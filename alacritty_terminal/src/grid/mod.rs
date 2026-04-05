//! A specialized 2D grid implementation optimized for use in a terminal.

use std::cmp::{max, min};
use std::ops::{Bound, Deref, Index, IndexMut, Range, RangeBounds};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use log::debug;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::index::{Column, Line, Point};
use crate::term::cell::{Cell, Flags, ResetDiscriminant};
use crate::vte::ansi::{CharsetIndex, StandardCharset};

pub mod compact;
pub mod resize;
mod row;
mod storage;
#[cfg(test)]
mod tests;

pub use self::compact::CompactRow;
pub use self::row::Row;
use self::storage::Storage;

pub trait GridCell: Sized {
    /// Check if the cell contains any content.
    fn is_empty(&self) -> bool;

    /// Perform an opinionated cell reset based on a template cell.
    fn reset(&mut self, template: &Self);

    fn flags(&self) -> &Flags;
    fn flags_mut(&mut self) -> &mut Flags;
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Cursor<T> {
    /// The location of this cursor.
    pub point: Point,

    /// Template cell when using this cursor.
    pub template: T,

    /// Currently configured graphic character sets.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub charsets: Charsets,

    /// Tracks if the next call to input will need to first handle wrapping.
    ///
    /// This is true after the last column is set with the input function. Any function that
    /// implicitly sets the line or column needs to set this to false to avoid wrapping twice.
    ///
    /// Tracking `input_needs_wrap` makes it possible to not store a cursor position that exceeds
    /// the number of columns, which would lead to index out of bounds when interacting with arrays
    /// without sanitization.
    pub input_needs_wrap: bool,
}

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub struct Charsets([StandardCharset; 4]);

impl Index<CharsetIndex> for Charsets {
    type Output = StandardCharset;

    fn index(&self, index: CharsetIndex) -> &StandardCharset {
        &self.0[index as usize]
    }
}

impl IndexMut<CharsetIndex> for Charsets {
    fn index_mut(&mut self, index: CharsetIndex) -> &mut StandardCharset {
        &mut self.0[index as usize]
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Scroll {
    Delta(i32),
    PageUp,
    PageDown,
    Top,
    Bottom,
}

/// Grid based terminal content storage.
///
/// ```notrust
/// ┌─────────────────────────┐  <-- max_scroll_limit + lines
/// │                         │
/// │      UNINITIALIZED      │
/// │                         │
/// ├─────────────────────────┤  <-- self.raw.inner.len()
/// │                         │
/// │      RESIZE BUFFER      │
/// │                         │
/// ├─────────────────────────┤  <-- self.history_size() + lines
/// │                         │
/// │     SCROLLUP REGION     │
/// │                         │
/// ├─────────────────────────┤v lines
/// │                         │|
/// │     VISIBLE  REGION     │|
/// │                         │|
/// ├─────────────────────────┤^ <-- display_offset
/// │                         │
/// │    SCROLLDOWN REGION    │
/// │                         │
/// └─────────────────────────┘  <-- zero
///                           ^
///                        columns
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Grid<T> {
    /// Current cursor for writing data.
    #[cfg_attr(feature = "serde", serde(default))]
    pub cursor: Cursor<T>,

    /// Last saved cursor.
    #[cfg_attr(feature = "serde", serde(default))]
    pub saved_cursor: Cursor<T>,

    /// Lines in the grid. Each row holds a list of cells corresponding to the
    /// columns in that row.
    raw: Storage<T>,

    /// Number of columns.
    columns: usize,

    /// Number of visible lines.
    lines: usize,

    /// Offset of displayed area.
    ///
    /// If the displayed region isn't at the bottom of the screen, it stays
    /// stationary while more text is emitted. The scrolling implementation
    /// updates this offset accordingly.
    display_offset: usize,

    /// Maximum number of lines in history.
    max_scroll_limit: usize,

    /// Compressed scrollback rows, ordered oldest-first.
    ///
    /// When scrollback exceeds a threshold, old history rows are compressed
    /// into this vec to reduce memory usage. They are decompressed back into
    /// the ring buffer on demand (e.g. when the user scrolls up).
    #[cfg_attr(feature = "serde", serde(default, skip))]
    compressed_history: Vec<CompactRow>,

    /// Rate-limited logging state for scrollback compression activity.
    #[cfg_attr(feature = "serde", serde(skip))]
    scrollback_stats: ScrollbackStats,
}

/// Tracks compression/decompression activity for periodic debug logging.
///
/// Accumulates counts and byte estimates between log emissions, then resets.
/// Emits at most once per second as long as there is activity.
#[derive(Debug, Clone)]
struct ScrollbackStats {
    last_log: Instant,
    rows_compressed: usize,
    rows_thawed: usize,
    rows_resized_on_thaw: usize,
    resize_from_cols: usize,
    resize_to_cols: usize,
}

static SCROLLBACK_LOGGING_ENABLED: AtomicBool = AtomicBool::new(true);

impl Default for ScrollbackStats {
    fn default() -> Self {
        Self {
            last_log: Instant::now(),
            rows_compressed: 0,
            rows_thawed: 0,
            rows_resized_on_thaw: 0,
            resize_from_cols: 0,
            resize_to_cols: 0,
        }
    }
}

impl ScrollbackStats {
    /// Record that rows were compressed.
    fn record_compress(&mut self, count: usize) {
        self.rows_compressed += count;
    }

    /// Record that rows were thawed, with optional resize info.
    fn record_thaw(&mut self, count: usize, resized: usize, from_cols: usize, to_cols: usize) {
        self.rows_thawed += count;
        self.rows_resized_on_thaw += resized;
        if resized > 0 {
            self.resize_from_cols = from_cols;
            self.resize_to_cols = to_cols;
        }
    }

    /// Emit a debug log line if at least 1 second has elapsed and there's activity.
    fn maybe_log(&mut self, hot: usize, compressed: usize, columns: usize, compressed_bytes: usize) {
        if !SCROLLBACK_LOGGING_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        if self.rows_compressed == 0 && self.rows_thawed == 0 {
            return;
        }
        if self.last_log.elapsed().as_secs() < 1 {
            return;
        }

        let total = hot + compressed;
        let hot_bytes_approx = hot * columns * 24;
        let compressed_mib = compressed_bytes as f64 / (1024.0 * 1024.0);
        let hot_mib = hot_bytes_approx as f64 / (1024.0 * 1024.0);
        let saved_bytes = (compressed * columns * 24).saturating_sub(compressed_bytes);
        let saved_mib = saved_bytes as f64 / (1024.0 * 1024.0);
        let ratio = if compressed_bytes > 0 {
            (compressed * columns * 24) as f64 / compressed_bytes as f64
        } else {
            0.0
        };

        if self.rows_compressed > 0 && self.rows_thawed > 0 {
            debug!(
                "[scrollback] compressed {} rows, thawed {} rows ({} resized {} to {} cols) \
                 hot={} compressed={} total={} compressed_mem={:.1} MiB hot_mem=~{:.1} MiB \
                 saved={:.1} MiB ratio={:.1}:1",
                self.rows_compressed, self.rows_thawed,
                self.rows_resized_on_thaw, self.resize_from_cols, self.resize_to_cols,
                hot, compressed, total, compressed_mib, hot_mib, saved_mib, ratio,
            );
        } else if self.rows_compressed > 0 {
            debug!(
                "[scrollback] compressed {} rows hot={} compressed={} total={} \
                 compressed_mem={:.1} MiB hot_mem=~{:.1} MiB saved={:.1} MiB ratio={:.1}:1",
                self.rows_compressed,
                hot, compressed, total, compressed_mib, hot_mib, saved_mib, ratio,
            );
        } else {
            debug!(
                "[scrollback] thawed {} rows ({} resized {} to {} cols) \
                 hot={} compressed={} total={} compressed_mem={:.1} MiB hot_mem=~{:.1} MiB",
                self.rows_thawed,
                self.rows_resized_on_thaw, self.resize_from_cols, self.resize_to_cols,
                hot, compressed, total, compressed_mib, hot_mib,
            );
        }

        self.rows_compressed = 0;
        self.rows_thawed = 0;
        self.rows_resized_on_thaw = 0;
        self.resize_from_cols = 0;
        self.resize_to_cols = 0;
        self.last_log = Instant::now();
    }
}

/// Enable or disable scrollback compression debug logging at runtime.
pub fn set_scrollback_logging(enabled: bool) {
    SCROLLBACK_LOGGING_ENABLED.store(enabled, Ordering::Relaxed);
}

impl<T: GridCell + Default + PartialEq> Grid<T> {
    pub fn new(lines: usize, columns: usize, max_scroll_limit: usize) -> Grid<T> {
        Grid {
            raw: Storage::with_capacity(lines, columns),
            max_scroll_limit,
            display_offset: 0,
            saved_cursor: Cursor::default(),
            cursor: Cursor::default(),
            scrollback_stats: ScrollbackStats::default(),
            lines,
            columns,
            compressed_history: Vec::new(),
        }
    }

    /// Update the size of the scrollback history.
    pub fn update_history(&mut self, history_size: usize) {
        let current_history_size = self.history_size();
        if current_history_size > history_size {
            self.raw.shrink_lines(current_history_size - history_size);
        }
        self.display_offset = min(self.display_offset, history_size);
        self.max_scroll_limit = history_size;
    }

    pub fn scroll_display(&mut self, scroll: Scroll) {
        self.display_offset = match scroll {
            Scroll::Delta(count) => {
                min(max((self.display_offset as i32) + count, 0) as usize, self.history_size())
            },
            Scroll::PageUp => min(self.display_offset + self.lines, self.history_size()),
            Scroll::PageDown => self.display_offset.saturating_sub(self.lines),
            Scroll::Top => self.history_size(),
            Scroll::Bottom => 0,
        };
    }

    fn increase_scroll_limit(&mut self, count: usize) {
        let count = min(count, self.max_scroll_limit - self.history_size());
        if count != 0 {
            self.raw.initialize(count, self.columns);
        }
    }

    fn decrease_scroll_limit(&mut self, count: usize) {
        let count = min(count, self.history_size());
        if count != 0 {
            self.raw.shrink_lines(min(count, self.history_size()));
            self.display_offset = min(self.display_offset, self.history_size());
        }
    }

    #[inline]
    pub fn scroll_down<D>(&mut self, region: &Range<Line>, positions: usize)
    where
        T: ResetDiscriminant<D>,
        D: PartialEq,
    {
        // When rotating the entire region, just reset everything.
        if region.end - region.start <= positions {
            for i in (region.start.0..region.end.0).map(Line::from) {
                self.raw[i].reset(&self.cursor.template);
            }

            return;
        }

        // Which implementation we can use depends on the existence of a scrollback history.
        //
        // Since a scrollback history prevents us from rotating the entire buffer downwards, we
        // instead have to rely on a slower, swap-based implementation.
        if self.max_scroll_limit == 0 {
            // Swap the lines fixed at the bottom to their target positions after rotation.
            //
            // Since we've made sure that the rotation will never rotate away the entire region, we
            // know that the position of the fixed lines before the rotation must already be
            // visible.
            //
            // We need to start from the top, to make sure the fixed lines aren't swapped with each
            // other.
            let screen_lines = self.screen_lines() as i32;
            for i in (region.end.0..screen_lines).map(Line::from) {
                self.raw.swap(i, i - positions as i32);
            }

            // Rotate the entire line buffer downward.
            self.raw.rotate_down(positions);

            // Ensure all new lines are fully cleared.
            for i in (0..positions).map(Line::from) {
                self.raw[i].reset(&self.cursor.template);
            }

            // Swap the fixed lines at the top back into position.
            for i in (0..region.start.0).map(Line::from) {
                self.raw.swap(i, i + positions);
            }
        } else {
            // Subregion rotation.
            let range = (region.start + positions).0..region.end.0;
            for line in range.rev().map(Line::from) {
                self.raw.swap(line, line - positions);
            }

            let range = region.start.0..(region.start + positions).0;
            for line in range.rev().map(Line::from) {
                self.raw[line].reset(&self.cursor.template);
            }
        }
    }

    /// Move lines at the bottom toward the top.
    ///
    /// This is the performance-sensitive part of scrolling.
    pub fn scroll_up<D>(&mut self, region: &Range<Line>, positions: usize)
    where
        T: ResetDiscriminant<D>,
        D: PartialEq,
    {
        // When rotating the entire region with fixed lines at the top, just reset everything.
        if region.end - region.start <= positions && region.start != 0 {
            for i in (region.start.0..region.end.0).map(Line::from) {
                self.raw[i].reset(&self.cursor.template);
            }

            return;
        }

        // Update display offset when not pinned to active area.
        if self.display_offset != 0 {
            self.display_offset = min(self.display_offset + positions, self.max_scroll_limit);
        }

        // Only rotate the entire history if the active region starts at the top.
        if region.start == 0 {
            // Create scrollback for the new lines.
            self.increase_scroll_limit(positions);

            // Swap the lines fixed at the top to their target positions after rotation.
            //
            // Since we've made sure that the rotation will never rotate away the entire region, we
            // know that the position of the fixed lines before the rotation must already be
            // visible.
            //
            // We need to start from the bottom, to make sure the fixed lines aren't swapped with
            // each other.
            for i in (0..region.start.0).rev().map(Line::from) {
                self.raw.swap(i, i + positions);
            }

            // Rotate the entire line buffer upward.
            self.raw.rotate(-(positions as isize));

            // Swap the fixed lines at the bottom back into position.
            let screen_lines = self.screen_lines() as i32;
            for i in (region.end.0..screen_lines).rev().map(Line::from) {
                self.raw.swap(i, i - positions);
            }
        } else {
            // Rotate lines without moving anything into history.
            for i in (region.start.0..region.end.0 - positions as i32).map(Line::from) {
                self.raw.swap(i, i + positions);
            }
        }

        // Ensure all new lines are fully cleared.
        for i in (region.end.0 - positions as i32..region.end.0).map(Line::from) {
            self.raw[i].reset(&self.cursor.template);
        }
    }

    pub fn clear_viewport<D>(&mut self)
    where
        T: ResetDiscriminant<D>,
        D: PartialEq,
    {
        // Determine how many lines to scroll up by.
        let end = Point::new(Line(self.lines as i32 - 1), Column(self.columns()));
        let mut iter = self.iter_from(end);
        while let Some(cell) = iter.prev() {
            if !cell.is_empty() || cell.point.line < 0 {
                break;
            }
        }
        debug_assert!(iter.point.line >= -1);
        let positions = (iter.point.line.0 + 1) as usize;
        let region = Line(0)..Line(self.lines as i32);

        // Clear the viewport.
        self.scroll_up(&region, positions);

        // Reset rotated lines.
        for line in (0..(self.lines - positions)).map(Line::from) {
            self.raw[line].reset(&self.cursor.template);
        }
    }

    /// Completely reset the grid state.
    pub fn reset<D>(&mut self)
    where
        T: ResetDiscriminant<D>,
        D: PartialEq,
    {
        self.clear_history();

        self.saved_cursor = Cursor::default();
        self.cursor = Cursor::default();
        self.display_offset = 0;

        // Reset all visible lines.
        let range = self.topmost_line().0..(self.screen_lines() as i32);
        for line in range.map(Line::from) {
            self.raw[line].reset(&self.cursor.template);
        }
    }
}

impl<T> Grid<T> {
    /// Reset a visible region within the grid.
    pub fn reset_region<D, R: RangeBounds<Line>>(&mut self, bounds: R)
    where
        T: ResetDiscriminant<D> + GridCell + Default,
        D: PartialEq,
    {
        let start = match bounds.start_bound() {
            Bound::Included(line) => *line,
            Bound::Excluded(line) => *line + 1,
            Bound::Unbounded => Line(0),
        };

        let end = match bounds.end_bound() {
            Bound::Included(line) => *line + 1,
            Bound::Excluded(line) => *line,
            Bound::Unbounded => Line(self.screen_lines() as i32),
        };

        debug_assert!(start < self.screen_lines() as i32);
        debug_assert!(end <= self.screen_lines() as i32);

        for line in (start.0..end.0).map(Line::from) {
            self.raw[line].reset(&self.cursor.template);
        }
    }

    #[inline]
    pub fn clear_history(&mut self) {
        // Explicitly purge all lines from history.
        self.raw.shrink_lines(self.history_size());

        // Also clear any compressed scrollback.
        self.compressed_history.clear();

        // Reset display offset.
        self.display_offset = 0;
    }

    /// This is used only for initializing after loading ref-tests.
    #[inline]
    pub fn initialize_all(&mut self)
    where
        T: GridCell + Default,
    {
        // Remove all cached lines to clear them of any content.
        self.truncate();

        // Initialize everything with empty new lines.
        self.raw.initialize(self.max_scroll_limit - self.history_size(), self.columns);
    }

    /// This is used only for truncating before saving ref-tests.
    #[inline]
    pub fn truncate(&mut self) {
        self.raw.truncate();
    }

    /// Iterate over all cells in the grid starting at a specific point.
    #[inline]
    pub fn iter_from(&self, point: Point) -> GridIterator<'_, T> {
        let end = Point::new(self.bottommost_line(), self.last_column());
        GridIterator { grid: self, point, end }
    }

    /// Iterate over all visible cells.
    ///
    /// This is slightly more optimized than calling `Grid::iter_from` in combination with
    /// `Iterator::take_while`.
    #[inline]
    pub fn display_iter(&self) -> GridIterator<'_, T> {
        let last_column = self.last_column();
        let start = Point::new(Line(-(self.display_offset() as i32) - 1), last_column);
        let end_line = min(start.line + self.screen_lines(), self.bottommost_line());
        let end = Point::new(end_line, last_column);

        GridIterator { grid: self, point: start, end }
    }

    #[inline]
    pub fn display_offset(&self) -> usize {
        self.display_offset
    }

    #[inline]
    pub fn cursor_cell(&mut self) -> &mut T {
        let point = self.cursor.point;
        let cols = self.columns;
        let lines = self.lines;
        let history = self.history_size();
        let topmost = self.topmost_line();
        let bottommost = self.bottommost_line();
        let display_offset = self.display_offset;
        let compressed = self.compressed_history.len();
        let input_needs_wrap = self.cursor.input_needs_wrap;
        let raw_len = self.raw.len();

        if point.column.0 >= cols
            || point.line < topmost
            || point.line > bottommost
        {
            panic!(
                "cursor_cell() out of bounds.\n\
                 cursor point:      {:?}\n\
                 input_needs_wrap:   {}\n\
                 grid columns:       {}\n\
                 grid screen_lines:  {}\n\
                 history_size:       {}\n\
                 raw.len:            {}\n\
                 topmost_line:       {:?}\n\
                 bottommost_line:    {:?}\n\
                 display_offset:     {}\n\
                 compressed_history: {}",
                point,
                input_needs_wrap,
                cols,
                lines,
                history,
                raw_len,
                topmost,
                bottommost,
                display_offset,
                compressed,
            );
        }

        &mut self[point.line][point.column]
    }
}

impl<T: PartialEq> PartialEq for Grid<T> {
    fn eq(&self, other: &Self) -> bool {
        // Compare struct fields and check result of grid comparison.
        self.raw.eq(&other.raw)
            && self.columns.eq(&other.columns)
            && self.lines.eq(&other.lines)
            && self.display_offset.eq(&other.display_offset)
            && self.compressed_history.eq(&other.compressed_history)
    }
}

impl<T> Index<Line> for Grid<T> {
    type Output = Row<T>;

    #[inline]
    fn index(&self, index: Line) -> &Row<T> {
        &self.raw[index]
    }
}

impl<T> IndexMut<Line> for Grid<T> {
    #[inline]
    fn index_mut(&mut self, index: Line) -> &mut Row<T> {
        &mut self.raw[index]
    }
}

impl<T> Index<Point> for Grid<T> {
    type Output = T;

    #[inline]
    fn index(&self, point: Point) -> &T {
        &self[point.line][point.column]
    }
}

impl<T> IndexMut<Point> for Grid<T> {
    #[inline]
    fn index_mut(&mut self, point: Point) -> &mut T {
        &mut self[point.line][point.column]
    }
}

// Grid dimensions.
// ---------------------------------------------------------------------------
// Cell-specific scrollback compression
// ---------------------------------------------------------------------------

impl Grid<Cell> {
    /// Compress old scrollback rows to reduce memory usage.
    ///
    /// Rows beyond `keep_hot` lines from the visible area are compressed into
    /// `CompactRow` form and removed from the ring buffer. This can reduce
    /// scrollback memory by 10–40× for typical terminal output.
    ///
    /// After compression, `history_size()` (hot rows only) decreases, but
    /// `total_history_size()` stays the same.
    pub fn compress_old_scrollback(&mut self, keep_hot: usize) {
        let hot_history = self.history_size();
        if hot_history <= keep_hot {
            return;
        }
        let to_compress = hot_history - keep_hot;

        // Compress the oldest rows first (farthest from visible area).
        for i in 0..to_compress {
            let line_idx = Line(-((hot_history - i) as i32));
            let compact = CompactRow::compress(&self.raw[line_idx]);
            self.compressed_history.push(compact);
        }

        // Remove compressed rows from the ring buffer.
        self.raw.shrink_lines(to_compress);
        self.display_offset = min(self.display_offset, self.history_size());

        self.scrollback_stats.record_compress(to_compress);
        let compressed_bytes = self.compressed_history_bytes();
        self.scrollback_stats.maybe_log(
            self.history_size(),
            self.compressed_history.len(),
            self.columns,
            compressed_bytes,
        );
    }

    /// Decompress the newest N compressed history rows back into the ring
    /// buffer so they become accessible via normal `Line` indexing.
    pub fn thaw_compressed_history(&mut self, count: usize) {
        let count = count.min(self.compressed_history.len());
        if count == 0 {
            return;
        }

        // Make room in the ring buffer for the decompressed rows.
        self.raw.initialize(count, self.columns);

        let mut resized = 0usize;
        let mut last_old_cols = 0usize;

        // The new oldest slots are at the far end of history.
        let new_history = self.history_size();
        for i in 0..count {
            let compressed_idx = self.compressed_history.len() - count + i;
            let mut decompressed = self.compressed_history[compressed_idx].decompress();

            // Compressed rows may predate a column resize, so reconcile widths.
            let old_cols = decompressed.len();
            if decompressed.len() < self.columns {
                decompressed.grow(self.columns);
                resized += 1;
                last_old_cols = old_cols;
            } else if decompressed.len() > self.columns {
                decompressed.shrink(self.columns);
                resized += 1;
                last_old_cols = old_cols;
            }

            let line_idx = Line(-((new_history - i) as i32));
            self.raw[line_idx] = decompressed;
        }

        // Remove from compressed storage.
        self.compressed_history.truncate(self.compressed_history.len() - count);

        self.scrollback_stats.record_thaw(count, resized, last_old_cols, self.columns);
        let compressed_bytes = self.compressed_history_bytes();
        self.scrollback_stats.maybe_log(
            self.history_size(),
            self.compressed_history.len(),
            self.columns,
            compressed_bytes,
        );
    }

    /// Automatically compress old scrollback if hot history exceeds the threshold.
    /// Call this after operations that grow scrollback (e.g. scroll_up).
    /// The threshold is 2× the visible screen lines — recent history stays hot
    /// for fast scrolling, older history gets compressed.
    pub fn compact_scrollback_if_needed(&mut self) {
        let threshold = self.lines * 2;
        let keep_hot = max(threshold, self.display_offset + self.lines);
        if self.history_size() > keep_hot {
            let max_per_call = 500;
            let excess = self.history_size() - keep_hot;
            let batch = min(excess, max_per_call);
            self.compress_old_scrollback(self.history_size() - batch);
        }
    }

    /// Scroll the display, automatically thawing compressed rows if needed.
    /// This is the Cell-specific version that handles compressed history.
    pub fn scroll_display_with_thaw(&mut self, scroll: Scroll) {
        let total_history = self.total_history_size();
        let new_offset = match scroll {
            Scroll::Delta(count) => {
                min(max((self.display_offset as i32) + count, 0) as usize, total_history)
            },
            Scroll::PageUp => min(self.display_offset + self.lines, total_history),
            Scroll::PageDown => self.display_offset.saturating_sub(self.lines),
            Scroll::Top => total_history,
            Scroll::Bottom => 0,
        };

        let hot_history = self.history_size();
        if new_offset > hot_history {
            let needed = new_offset - hot_history;
            self.thaw_compressed_history(needed);
        }

        self.scroll_display(scroll);
    }

    /// Number of compressed history rows.
    pub fn compressed_history_len(&self) -> usize {
        self.compressed_history.len()
    }

    /// Total history size including both hot and compressed rows.
    pub fn total_history_size(&self) -> usize {
        self.history_size() + self.compressed_history.len()
    }

    /// Approximate heap memory used by compressed history, in bytes.
    pub fn compressed_history_bytes(&self) -> usize {
        self.compressed_history
            .iter()
            .map(|r| r.heap_bytes() + std::mem::size_of::<CompactRow>())
            .sum()
    }

    /// Line farthest up in the grid including compressed history.
    /// Use this instead of `topmost_line()` when the caller can thaw on demand.
    pub fn total_topmost_line(&self) -> Line {
        Line(-(self.total_history_size() as i32))
    }
}

pub trait Dimensions {
    /// Total number of lines in the buffer, this includes scrollback and visible lines.
    fn total_lines(&self) -> usize;

    /// Height of the viewport in lines.
    fn screen_lines(&self) -> usize;

    /// Width of the terminal in columns.
    fn columns(&self) -> usize;

    /// Index for the last column.
    #[inline]
    fn last_column(&self) -> Column {
        Column(self.columns() - 1)
    }

    /// Line farthest up in the grid history.
    #[inline]
    fn topmost_line(&self) -> Line {
        Line(-(self.history_size() as i32))
    }

    /// Line farthest down in the grid history.
    #[inline]
    fn bottommost_line(&self) -> Line {
        Line(self.screen_lines() as i32 - 1)
    }

    /// Number of invisible lines part of the scrollback history.
    #[inline]
    fn history_size(&self) -> usize {
        self.total_lines().saturating_sub(self.screen_lines())
    }
}

impl<G> Dimensions for Grid<G> {
    #[inline]
    fn total_lines(&self) -> usize {
        self.raw.len()
    }

    #[inline]
    fn screen_lines(&self) -> usize {
        self.lines
    }

    #[inline]
    fn columns(&self) -> usize {
        self.columns
    }
}

#[cfg(test)]
impl Dimensions for (usize, usize) {
    fn total_lines(&self) -> usize {
        self.0
    }

    fn screen_lines(&self) -> usize {
        self.0
    }

    fn columns(&self) -> usize {
        self.1
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Indexed<T> {
    pub point: Point,
    pub cell: T,
}

impl<T> Deref for Indexed<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.cell
    }
}

/// Grid cell iterator.
pub struct GridIterator<'a, T> {
    /// Immutable grid reference.
    grid: &'a Grid<T>,

    /// Current position of the iterator within the grid.
    point: Point,

    /// Last cell included in the iterator.
    end: Point,
}

impl<'a, T> GridIterator<'a, T> {
    /// Current iterator position.
    pub fn point(&self) -> Point {
        self.point
    }

    /// Cell at the current iterator position.
    pub fn cell(&self) -> &'a T {
        &self.grid[self.point]
    }
}

impl<'a, T> Iterator for GridIterator<'a, T> {
    type Item = Indexed<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Stop once we've reached the end of the grid.
        if self.point >= self.end {
            return None;
        }

        match self.point {
            Point { column, .. } if column >= self.grid.last_column() => {
                self.point.column = Column(0);
                self.point.line += 1;
            },
            _ => self.point.column += Column(1),
        }

        // Diagnostic guard: verify the point is within grid bounds before
        // indexing. If this fires, the log line contains the state needed
        // to write a reproduction test for the OOB crash.
        let lines_in_buffer = self.grid.total_lines();
        let history = self.grid.history_size();
        let screen = self.grid.screen_lines();
        let cols = self.grid.columns();
        let topmost = self.grid.topmost_line();
        let bottommost = self.grid.bottommost_line();

        if self.point.line < topmost
            || self.point.line > bottommost
            || self.point.column.0 >= cols
        {
            panic!(
                "GridIterator::next() about to access out-of-bounds point.\n\
                 point:          {:?}\n\
                 end:            {:?}\n\
                 grid lines:     {} (screen={}, history={}, buffer={})\n\
                 grid columns:   {}\n\
                 topmost_line:   {:?}\n\
                 bottommost_line:{:?}\n\
                 display_offset: {}",
                self.point,
                self.end,
                screen + history,
                screen,
                history,
                lines_in_buffer,
                cols,
                topmost,
                bottommost,
                self.grid.display_offset(),
            );
        }

        Some(Indexed { cell: &self.grid[self.point], point: self.point })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.point >= self.end {
            return (0, Some(0));
        }

        let size = if self.point.line == self.end.line {
            (self.end.column - self.point.column).0
        } else {
            let cols_on_first_line = self.grid.columns.saturating_sub(self.point.column.0 + 1);
            let middle_lines = (self.end.line - self.point.line).0 as usize - 1;
            let cols_on_last_line = self.end.column + 1;

            cols_on_first_line + middle_lines * self.grid.columns + cols_on_last_line.0
        };

        (size, Some(size))
    }
}

/// Bidirectional iterator.
pub trait BidirectionalIterator: Iterator {
    fn prev(&mut self) -> Option<Self::Item>;
}

impl<T> BidirectionalIterator for GridIterator<'_, T> {
    fn prev(&mut self) -> Option<Self::Item> {
        let topmost_line = self.grid.topmost_line();
        let last_column = self.grid.last_column();

        // Stop once we've reached the end of the grid.
        if self.point <= Point::new(topmost_line, Column(0)) {
            return None;
        }

        match self.point {
            Point { column: Column(0), .. } => {
                self.point.column = last_column;
                self.point.line -= 1;
            },
            _ => self.point.column -= Column(1),
        }

        Some(Indexed { cell: &self.grid[self.point], point: self.point })
    }
}
