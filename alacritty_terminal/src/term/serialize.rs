//! Terminal state snapshot and restore via serde.
//!
//! Captures the full terminal state — both grid buffers, cursor, mode flags,
//! and scroll region — as a [`TermState`] struct that can be serialised with
//! any serde-compatible format (bincode, JSON, …) and later applied back to
//! a [`Term`] via [`Term::restore`].
//!
//! [`Term`]: crate::Term

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::grid::Grid;
use crate::index::Line;
use crate::term::cell::Cell;
use crate::term::TermMode;

/// Complete terminal state for serialisation and deserialisation.
///
/// Captures everything needed to restore a terminal session after reconnect:
/// both grid buffers (active and inactive, covering alternate screen), cursor
/// position and template, terminal mode flags, and scroll region.
///
/// # What is captured
///
/// | Field | Purpose |
/// |---|---|
/// | `grid` | Active grid with cursor, scrollback, cell content & attributes |
/// | `inactive_grid` | Inactive buffer (primary when alt screen is active, or vice versa) |
/// | `mode_bits` | `TermMode` bitflags: bracketed paste, mouse mode, alt screen, etc. |
/// | `scroll_region` | DECSTBM scroll region (top..bottom viewport lines) |
///
/// # What is NOT captured (defaults on restore)
///
/// - Character set mappings (`Charsets`) — almost never non-default
/// - Tab stops — reconstructed as standard 8-column stops
/// - Window title / title stack
/// - Keyboard mode stack
/// - Terminal colour overrides (come from client config)
/// - Cursor style (comes from client config)
/// - Selection state (client-side UI)
/// - Vi mode cursor (client-side UI)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TermState {
    /// Active grid (primary or alternate, depending on mode).
    pub grid: Grid<Cell>,

    /// Inactive grid (the other buffer).
    pub inactive_grid: Grid<Cell>,

    /// Terminal mode flags as raw `TermMode` bits.
    ///
    /// Stored as `u32` because `TermMode` (a `bitflags!` type) may not have
    /// serde derives. Use [`TermState::mode()`] to get the typed value.
    pub(crate) mode_bits: u32,

    /// Scroll region (top..bottom viewport lines).
    pub scroll_region: Range<Line>,
}

impl TermState {
    /// Terminal mode flags.
    pub fn mode(&self) -> TermMode {
        TermMode::from_bits_truncate(self.mode_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::TermState;
    use crate::event::VoidListener;
    use crate::grid::{Dimensions, Grid};
    use crate::index::{Column, Line};
    use crate::term::cell::{Cell, Flags};
    use crate::term::test::TermSize;
    use crate::term::{Config, Term, TermMode};
    use crate::vte::ansi;

    /// Helper: create a term, push bytes through the parser, return the term.
    fn term_with(cols: usize, rows: usize, input: &[u8]) -> Term<VoidListener> {
        let size = TermSize::new(cols, rows);
        let mut term = Term::new(Config::default(), &size, VoidListener);
        let mut parser: ansi::Processor = ansi::Processor::new();
        parser.advance(&mut term, input);
        term
    }

    /// Helper: extract visible text (screen only, no scrollback) trimmed.
    fn visible_text(term: &Term<VoidListener>) -> String {
        let grid = term.grid();
        let mut lines = Vec::new();
        for row_idx in 0..grid.screen_lines() {
            let line = Line(row_idx as i32);
            let mut s = String::new();
            for col_idx in 0..grid.columns() {
                let col = Column(col_idx);
                let cell = &grid[line][col];
                if cell.flags.contains(Flags::WIDE_CHAR_SPACER) {
                    continue;
                }
                s.push(cell.c);
            }
            lines.push(s.trim_end().to_string());
        }
        while lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }
        lines.join("\n")
    }

    /// Helper: compare per-cell attributes for the visible screen area.
    fn assert_cells_equal(a: &Term<VoidListener>, b: &Term<VoidListener>) {
        let ga = a.grid();
        let gb = b.grid();
        assert_eq!(ga.screen_lines(), gb.screen_lines());
        assert_eq!(ga.columns(), gb.columns());

        for row_idx in 0..ga.screen_lines() {
            let line = Line(row_idx as i32);
            for col_idx in 0..ga.columns() {
                let col = Column(col_idx);
                let ca = &ga[line][col];
                let cb = &gb[line][col];
                assert_eq!(ca.c, cb.c, "char mismatch at ({row_idx}, {col_idx})");
                assert_eq!(ca.fg, cb.fg, "fg mismatch at ({row_idx}, {col_idx})");
                assert_eq!(ca.bg, cb.bg, "bg mismatch at ({row_idx}, {col_idx})");
                assert_eq!(ca.flags, cb.flags, "flags mismatch at ({row_idx}, {col_idx})");
            }
        }
    }

    #[test]
    fn grid_serde_round_trip() {
        let term = term_with(40, 10, b"\x1b[1;31mhello\x1b[0m world\r\nline two");
        let grid = term.grid();

        let json = serde_json::to_string(grid).expect("serialize Grid<Cell>");
        let grid2: Grid<Cell> = serde_json::from_str(&json).expect("deserialize Grid<Cell>");

        assert_eq!(grid.columns(), grid2.columns());
        assert_eq!(grid.screen_lines(), grid2.screen_lines());
        assert_eq!(grid.topmost_line(), grid2.topmost_line());

        for row_idx in 0..grid.screen_lines() {
            let line = Line(row_idx as i32);
            for col_idx in 0..grid.columns() {
                let col = Column(col_idx);
                let a = &grid[line][col];
                let b = &grid2[line][col];
                assert_eq!(a.c, b.c, "char mismatch at ({row_idx}, {col_idx})");
                assert_eq!(a.fg, b.fg, "fg mismatch at ({row_idx}, {col_idx})");
                assert_eq!(a.bg, b.bg, "bg mismatch at ({row_idx}, {col_idx})");
                assert_eq!(a.flags, b.flags, "flags mismatch at ({row_idx}, {col_idx})");
            }
        }
    }

    #[test]
    fn grid_serde_preserves_scrollback() {
        let term = term_with(
            40,
            4,
            b"line1\r\nline2\r\nline3\r\nline4\r\nline5\r\nline6\r\nline7\r\nline8",
        );
        let grid = term.grid();
        assert!(grid.topmost_line().0 < 0, "expected scrollback");

        let json = serde_json::to_string(grid).expect("serialize");
        let grid2: Grid<Cell> = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(grid.topmost_line(), grid2.topmost_line());

        for line_idx in grid.topmost_line().0..0 {
            let line = Line(line_idx);
            for col_idx in 0..grid.columns() {
                let col = Column(col_idx);
                assert_eq!(
                    grid[line][col].c,
                    grid2[line][col].c,
                    "scrollback mismatch at ({line_idx}, {col_idx})",
                );
            }
        }
    }

    #[test]
    fn grid_serde_preserves_wrapline() {
        let term = term_with(10, 4, b"abcdefghijklmno");
        let grid = term.grid();

        assert!(
            grid[Line(0)][Column(9)].flags.contains(Flags::WRAPLINE),
            "expected WRAPLINE on first row",
        );

        let json = serde_json::to_string(grid).expect("serialize");
        let grid2: Grid<Cell> = serde_json::from_str(&json).expect("deserialize");

        assert!(
            grid2[Line(0)][Column(9)].flags.contains(Flags::WRAPLINE),
            "WRAPLINE lost after serde round-trip",
        );
    }

    #[test]
    fn grid_serde_cursor_survives() {
        let term = term_with(40, 10, b"\x1b[1;31m\x1b[5;20Hhere");
        let grid = term.grid();
        assert_ne!(grid.cursor.point, Default::default());

        let json = serde_json::to_string(grid).expect("serialize");
        let grid2: Grid<Cell> = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(grid.cursor.point, grid2.cursor.point);
        assert_eq!(grid.cursor.template.fg, grid2.cursor.template.fg);
        assert_eq!(grid.cursor.template.bg, grid2.cursor.template.bg);
        assert_eq!(grid.cursor.template.flags, grid2.cursor.template.flags);
        assert_eq!(grid.cursor.input_needs_wrap, grid2.cursor.input_needs_wrap);
    }

    #[test]
    fn snapshot_restore_preserves_content_and_cursor() {
        let term1 = term_with(40, 10, b"\x1b[1;31mhello\x1b[0m world\r\n\x1b[5;20Hcursor here");
        let state = term1.snapshot();

        let size = TermSize::new(40, 10);
        let mut term2 = Term::new(Config::default(), &size, VoidListener);
        term2.restore(state);

        assert_eq!(visible_text(&term1), visible_text(&term2));
        assert_eq!(term1.grid().cursor.point, term2.grid().cursor.point);
        assert_cells_equal(&term1, &term2);
    }

    #[test]
    fn snapshot_restore_preserves_modes() {
        let term1 = term_with(40, 10, b"\x1b[?2004h\x1b[?1000hsome text");
        assert!(term1.mode().contains(TermMode::BRACKETED_PASTE));
        assert!(term1.mode().contains(TermMode::MOUSE_REPORT_CLICK));

        let state = term1.snapshot();
        let size = TermSize::new(40, 10);
        let mut term2 = Term::new(Config::default(), &size, VoidListener);
        term2.restore(state);

        assert_eq!(*term1.mode(), *term2.mode());
    }

    #[test]
    fn snapshot_restore_preserves_alternate_screen() {
        let mut input = Vec::new();
        input.extend_from_slice(b"primary line 1\r\nprimary line 2\r\n");
        input.extend_from_slice(b"\x1b[?1049h");
        input.extend_from_slice(b"alt screen content");

        let term1 = term_with(40, 10, &input);
        assert!(term1.mode().contains(TermMode::ALT_SCREEN));

        let state = term1.snapshot();
        let size = TermSize::new(40, 10);
        let mut term2 = Term::new(Config::default(), &size, VoidListener);
        term2.restore(state);

        assert!(term2.mode().contains(TermMode::ALT_SCREEN));
        assert_eq!(visible_text(&term1), visible_text(&term2));
    }

    #[test]
    fn snapshot_restore_preserves_scroll_region() {
        let term1 = term_with(40, 10, b"\x1b[3;8rtext after region set");
        let state1 = term1.snapshot();
        let scroll_region = state1.scroll_region.clone();
        assert_ne!(scroll_region, Line(0)..Line(10));

        let size = TermSize::new(40, 10);
        let mut term2 = Term::new(Config::default(), &size, VoidListener);
        term2.restore(state1);

        let state2 = term2.snapshot();
        assert_eq!(state2.scroll_region, scroll_region);
    }

    #[test]
    fn snapshot_restore_preserves_scrollback() {
        let term1 = term_with(
            40,
            4,
            b"line1\r\nline2\r\nline3\r\nline4\r\nline5\r\nline6\r\nline7\r\nline8",
        );
        assert!(term1.grid().topmost_line().0 < 0, "expected scrollback");

        let state = term1.snapshot();
        let size = TermSize::new(40, 4);
        let mut term2 = Term::new(Config::default(), &size, VoidListener);
        term2.restore(state);

        assert_eq!(term1.grid().topmost_line(), term2.grid().topmost_line());
        assert_eq!(visible_text(&term1), visible_text(&term2));

        for line_idx in term1.grid().topmost_line().0..0 {
            let line = Line(line_idx);
            for col_idx in 0..term1.grid().columns() {
                let col = Column(col_idx);
                assert_eq!(
                    term1.grid()[line][col].c,
                    term2.grid()[line][col].c,
                    "scrollback mismatch at ({line_idx}, {col_idx})",
                );
            }
        }
    }

    #[test]
    fn term_state_serde_json_round_trip() {
        let term = term_with(40, 10, b"\x1b[?2004h\x1b[1;31mhello\x1b[0m\x1b[5;20H");
        let state = term.snapshot();

        let json = serde_json::to_string(&state).expect("TermState JSON serialize");
        let state2: TermState = serde_json::from_str(&json).expect("TermState JSON deserialize");

        assert_eq!(state.grid.cursor.point, state2.grid.cursor.point);
        assert_eq!(state.mode(), state2.mode());
        assert_eq!(state.scroll_region, state2.scroll_region);
        assert_eq!(state.grid.columns(), state2.grid.columns());
        assert_eq!(state.grid.screen_lines(), state2.grid.screen_lines());
    }

    #[test]
    fn term_state_serde_bincode_round_trip() {
        let term = term_with(80, 24, b"\x1b[?2004h\x1b[?1000h\x1b[1;31mhello\x1b[0m world");
        let state = term.snapshot();

        let bytes = bincode::serde::encode_to_vec(&state, bincode::config::standard())
            .expect("TermState bincode serialize");
        let (state2, _): (TermState, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .expect("TermState bincode deserialize");

        assert_eq!(state.grid.cursor.point, state2.grid.cursor.point);
        assert_eq!(state.mode(), state2.mode());
        assert_eq!(state.scroll_region, state2.scroll_region);
        assert_eq!(state.grid.columns(), state2.grid.columns());
    }

    // -----------------------------------------------------------------------
    // Regression: Zed's pty-host restore path.
    //
    // Zed creates a Term with the current window dimensions, then calls
    // term.restore(snapshot) where the snapshot came from a pty-host that
    // may have had different dimensions and compressed history. The PTY
    // reader thread can start processing data before the UI thread's first
    // sync() resizes the grid to match the window.
    //
    // compressed_history has #[serde(skip)], so it is silently dropped.
    // This can leave display_offset > history_size() and/or the cursor
    // at a column that no longer exists.
    // -----------------------------------------------------------------------

    use crate::grid::Scroll;
    use crate::vte::ansi::Handler;

    /// Build a Term with real scrollback by pushing lines through the parser.
    fn term_with_scrollback(
        cols: usize,
        rows: usize,
        scrollback_lines: usize,
    ) -> Term<VoidListener> {
        let size = TermSize::new(cols, rows);
        let mut term = Term::new(Config::default(), &size, VoidListener);

        for n in 0..(rows + scrollback_lines) {
            let line = format!("line {n:>6}\r\n");
            let mut parser: ansi::Processor = ansi::Processor::new();
            parser.advance(&mut term, line.as_bytes());
        }

        term
    }

    /// Simulate Zed's from_pty_host path: create a fresh Term with window
    /// dimensions, serde round-trip the snapshot, then restore.
    fn zed_restore(
        snapshot_term: &Term<VoidListener>,
        window_cols: usize,
        window_rows: usize,
    ) -> Term<VoidListener> {
        let state = snapshot_term.snapshot();
        let bytes = serde_json::to_string(&state).expect("serialize");
        let state2: TermState = serde_json::from_str(&bytes).expect("deserialize");

        let size = TermSize::new(window_cols, window_rows);
        let mut restored = Term::new(Config::default(), &size, VoidListener);
        restored.restore(state2);
        restored
    }

    #[test]
    fn serde_drops_compressed_history() {
        let mut term = term_with_scrollback(80, 24, 200);
        assert!(term.grid().history_size() >= 200);

        term.grid_mut().compact_scrollback_if_needed();
        let compressed_before = term.grid().compressed_history_len();
        assert!(compressed_before > 0, "expected some compressed rows");
        let hot_before = term.grid().history_size();

        let state = term.snapshot();
        let json = serde_json::to_string(&state).expect("serialize");
        let state2: TermState = serde_json::from_str(&json).expect("deserialize");

        let size = TermSize::new(80, 24);
        let mut restored = Term::new(Config::default(), &size, VoidListener);
        restored.restore(state2);

        assert_eq!(restored.grid().compressed_history_len(), 0);
        assert_eq!(restored.grid().history_size(), hot_before);
    }

    #[test]
    fn restore_then_compact_clamps_display_offset() {
        // After restore, compact_scrollback_if_needed (called by sync) could
        // shrink history below display_offset if display_offset was preserved
        // from a larger history.
        let mut term = term_with_scrollback(80, 24, 200);

        // Scroll partway up.
        term.scroll_display(Scroll::Delta(50));
        assert_eq!(term.grid().display_offset(), 50);

        // Compact to create compressed history.
        term.grid_mut().compact_scrollback_if_needed();

        // Restore (compressed history dropped, but hot rows + display_offset kept).
        let mut restored = zed_restore(&term, 80, 24);

        // Simulate what sync() does: compact then iterate.
        restored.grid_mut().compact_scrollback_if_needed();

        assert!(
            restored.grid().display_offset() <= restored.grid().history_size(),
            "display_offset ({}) > history_size ({}) after restore + compact",
            restored.grid().display_offset(),
            restored.grid().history_size(),
        );

        let content = restored.renderable_content();
        let _count = content.display_iter.count();
        let _cursor_char = restored.grid()[content.cursor.point].c;
    }

    #[test]
    fn restore_same_dimensions_then_input() {
        // Pty-host snapshot with same dimensions as window. Cursor at last
        // column with input_needs_wrap. After restore, PTY reader calls input.
        let cols = 142;
        let rows = 24;
        let mut term = term_with_scrollback(cols, rows, 200);

        // Fill line to set input_needs_wrap.
        let fill: String = (0..cols).map(|i| (b'A' + (i % 26) as u8) as char).collect();
        let mut parser: ansi::Processor = ansi::Processor::new();
        parser.advance(&mut term, format!("\r{fill}").as_bytes());
        assert!(term.grid().cursor.input_needs_wrap);

        // Compact to create compressed history.
        term.grid_mut().compact_scrollback_if_needed();
        assert!(term.grid().compressed_history_len() > 0);

        // Restore into same-size term (no resize will be triggered).
        let mut restored = zed_restore(&term, cols, rows);

        // Cursor must be valid.
        assert!(
            restored.grid().cursor.point.column.0 < cols,
            "cursor column ({}) >= grid columns ({cols})",
            restored.grid().cursor.point.column.0,
        );

        // PTY reader sends data — must not panic.
        let mut parser2: ansi::Processor = ansi::Processor::new();
        parser2.advance(&mut restored, b"\r\n$ ");

        let content = restored.renderable_content();
        let _count = content.display_iter.count();
    }

    #[test]
    fn restore_different_dimensions_then_input() {
        // Pty-host had 143 columns, window now has 142. Cursor was at last
        // column (142) with input_needs_wrap. After restore the grid has 143
        // columns but no resize has happened yet. PTY data arrives.
        let snapshot_cols = 143;
        let window_cols = 142;
        let rows = 24;
        let mut term = term_with_scrollback(snapshot_cols, rows, 200);

        // Place cursor at last column of 143-col grid.
        let fill: String = (0..snapshot_cols)
            .map(|i| (b'A' + (i % 26) as u8) as char)
            .collect();
        let mut parser: ansi::Processor = ansi::Processor::new();
        parser.advance(&mut term, format!("\r{fill}").as_bytes());
        assert!(term.grid().cursor.input_needs_wrap);
        assert_eq!(term.grid().cursor.point.column, Column(snapshot_cols - 1));

        term.grid_mut().compact_scrollback_if_needed();

        // Restore into smaller window (grid still has snapshot dimensions
        // until the UI thread calls sync → resize).
        let mut restored = zed_restore(&term, window_cols, rows);

        // Grid has snapshot's column count after restore.
        assert_eq!(restored.grid().columns(), snapshot_cols);

        // PTY reader sends data BEFORE resize — grid is still 143 cols.
        // This should not panic.
        let mut parser2: ansi::Processor = ansi::Processor::new();
        parser2.advance(&mut restored, b"\r\n$ ");

        // Now simulate sync: resize to window dimensions, compact, iterate.
        restored.resize(TermSize::new(window_cols, rows));
        restored.grid_mut().compact_scrollback_if_needed();

        assert!(
            restored.grid().cursor.point.column.0 < window_cols,
            "cursor column ({}) >= grid columns ({window_cols}) after resize",
            restored.grid().cursor.point.column.0,
        );

        let content = restored.renderable_content();
        let _count = content.display_iter.count();
        let _cursor_char = restored.grid()[content.cursor.point].c;
    }

    #[test]
    fn restore_with_compressed_history_then_compact_reduces_below_offset() {
        // The critical scenario: snapshot was taken while scrolled up.
        // After restore, compressed history is gone, so hot history is
        // smaller. The FIRST compact_scrollback_if_needed on the restored
        // term could further reduce hot history below display_offset.
        let cols = 146;
        let rows = 24;
        let mut term = term_with_scrollback(cols, rows, 500);

        // Compact aggressively.
        term.grid_mut().compact_scrollback_if_needed();
        let hot = term.grid().history_size();
        let compressed = term.grid().compressed_history_len();
        assert!(compressed > 0);

        // Scroll up to near the edge of hot history.
        term.scroll_display(Scroll::Delta(hot as i32 - 1));
        let offset_before = term.grid().display_offset();
        assert!(offset_before > 0);

        // Restore — compressed rows are lost.
        let mut restored = zed_restore(&term, cols, rows);
        assert_eq!(restored.grid().compressed_history_len(), 0);

        // display_offset should not exceed available history.
        assert!(
            restored.grid().display_offset() <= restored.grid().history_size(),
            "display_offset ({}) > history_size ({}) immediately after restore",
            restored.grid().display_offset(),
            restored.grid().history_size(),
        );

        // Simulate sync: compact + renderable_content.
        restored.grid_mut().compact_scrollback_if_needed();
        assert!(
            restored.grid().display_offset() <= restored.grid().history_size(),
            "display_offset ({}) > history_size ({}) after restore + compact",
            restored.grid().display_offset(),
            restored.grid().history_size(),
        );

        let content = restored.renderable_content();
        let _count = content.display_iter.count();
        let _cursor_char = restored.grid()[content.cursor.point].c;
    }

    #[test]
    fn restore_idle_terminal_then_keypress() {
        // Matches the exact user scenario: terminal idle for a while (has
        // compressed history from repeated compact_scrollback_if_needed
        // calls during sync), then user presses Enter.
        let cols = 142;
        let rows = 24;
        let mut term = term_with_scrollback(cols, rows, 300);

        // Simulate idle: compact runs every sync frame while idle.
        for _ in 0..10 {
            term.grid_mut().compact_scrollback_if_needed();
        }
        assert!(term.grid().compressed_history_len() > 0);

        // Pty-host snapshots the idle terminal.
        let mut restored = zed_restore(&term, cols, rows);

        // Simulate more idle sync cycles on the restored term.
        for _ in 0..5 {
            restored.grid_mut().compact_scrollback_if_needed();
            let content = restored.renderable_content();
            let _count = content.display_iter.count();
        }

        // User presses Enter — shell outputs prompt on PTY reader thread.
        // This is the path that crashed: Term::input → cursor_cell().
        let mut parser: ansi::Processor = ansi::Processor::new();
        parser.advance(&mut restored, b"\r\n$ ");

        // Verify no panic in renderable_content after the input.
        let content = restored.renderable_content();
        let _count = content.display_iter.count();
        let _cursor_char = restored.grid()[content.cursor.point].c;
    }

    #[test]
    fn compressed_rows_not_resized_during_column_change() {
        // ROOT CAUSE: when the grid's column count changes (grow_columns /
        // shrink_columns), only hot rows in the ring buffer are rebuilt.
        // Compressed rows in `compressed_history` retain their old column
        // count. If those rows are later thawed (e.g. user scrolls up),
        // they have the WRONG number of columns. Any code that indexes
        // with `grid.columns` (clear_line, cursor_cell, display_iter)
        // panics because the row is shorter than expected.
        //
        // Crash: "range end index 171 out of range for slice of length 146"
        // in clear_line, or "the len is 146 but the index is 146" in
        // cursor_cell / GridIterator::next.
        let original_cols = 146;
        let rows = 24;
        let mut term = term_with_scrollback(original_cols, rows, 200);

        // Compact — oldest history rows are compressed with 146 columns.
        term.grid_mut().compact_scrollback_if_needed();
        let compressed = term.grid().compressed_history_len();
        assert!(compressed > 0, "need compressed rows for this test");

        // Grow columns to 171. Hot rows are rebuilt, but compressed rows
        // still store 146 columns internally.
        let new_cols = 171;
        term.resize(TermSize::new(new_cols, rows));
        assert_eq!(term.grid().columns(), new_cols);

        // Compressed rows still exist with the OLD column count.
        assert!(term.grid().compressed_history_len() > 0);

        // Thaw the compressed rows by scrolling to the top.
        term.grid_mut().scroll_display_with_thaw(Scroll::Top);

        // Verify ALL rows now have the correct column count.
        // If this fails, thawed rows have stale widths.
        let grid = term.grid();
        for line_idx in grid.topmost_line().0..grid.screen_lines() as i32 {
            let line = Line(line_idx);
            let row_len = grid[line].len();
            assert_eq!(
                row_len, new_cols,
                "row at line {line_idx} has {row_len} columns, expected {new_cols} \
                 (was a compressed row with old width)"
            );
        }

        // Exercise the code paths that crash:
        // 1. clear_line uses grid.columns() to build a range over the row.
        term.grid_mut().scroll_display(Scroll::Bottom);
        let mut parser: ansi::Processor = ansi::Processor::new();
        // CSI 2 K = clear entire line
        parser.advance(&mut term, b"\x1b[2K");

        // 2. renderable_content iterates all visible cells.
        let content = term.renderable_content();
        let _count = content.display_iter.count();

        // 3. cursor_cell accessed during input.
        parser.advance(&mut term, b"hello");
    }

    #[test]
    fn compressed_rows_not_resized_during_column_shrink() {
        // Same issue in reverse: compressed at 171 columns, then grid
        // shrinks to 146. Thawed rows are wider than grid.columns,
        // which is less immediately dangerous but still wrong.
        let original_cols = 171;
        let rows = 24;
        let mut term = term_with_scrollback(original_cols, rows, 200);

        term.grid_mut().compact_scrollback_if_needed();
        assert!(term.grid().compressed_history_len() > 0);

        // Shrink columns.
        let new_cols = 146;
        term.resize(TermSize::new(new_cols, rows));
        assert_eq!(term.grid().columns(), new_cols);

        // Thaw compressed rows (still have 171 columns).
        term.grid_mut().scroll_display_with_thaw(Scroll::Top);

        let grid = term.grid();
        for line_idx in grid.topmost_line().0..grid.screen_lines() as i32 {
            let line = Line(line_idx);
            let row_len = grid[line].len();
            assert_eq!(
                row_len, new_cols,
                "row at line {line_idx} has {row_len} columns, expected {new_cols}"
            );
        }
    }

    #[test]
    fn serde_restore_with_mismatched_compressed_row_widths() {
        // End-to-end reproduction of the pty-host crash path:
        // 1. Pty-host terminal starts at 146 columns, builds scrollback.
        // 2. Compaction compresses old rows (stored with 146 cols).
        // 3. Window resizes to 171 columns — hot rows rebuilt, compressed
        //    rows untouched.
        // 4. Pty-host takes snapshot (grid.columns=171, compressed rows
        //    have 146 cols but are serde(skip)'d).
        // 5. User scrolls up on restored terminal — thaw decompresses
        //    rows with 146 cols into a 171-col grid.
        // 6. clear_line / input / display_iter panics.
        let rows = 24;
        let mut term = term_with_scrollback(146, rows, 200);

        // Compress at 146 columns.
        term.grid_mut().compact_scrollback_if_needed();
        assert!(term.grid().compressed_history_len() > 0);

        // Resize to 171 columns (compressed rows still have 146).
        term.resize(TermSize::new(171, rows));

        // Snapshot and restore (compressed rows are dropped by serde).
        let mut restored = zed_restore(&term, 171, rows);
        assert_eq!(restored.grid().columns(), 171);
        // Compressed rows are gone after serde — the thaw path won't
        // trigger in this case, but verify the grid is consistent.

        // Now simulate the scenario where compressed rows WEREN'T dropped
        // (i.e., the pty-host thawed them before snapshot, producing rows
        // in the ring buffer with 146 columns in a 171-column grid).
        // We can do this by thawing on the original term, THEN snapshotting.
        term.grid_mut().scroll_display_with_thaw(Scroll::Top);
        // Now the ring buffer has thawed rows with 146 columns.
        let mut restored2 = zed_restore(&term, 171, rows);

        // These thawed-but-wrong-width rows are now in the ring buffer
        // and survived serde. Scrolling up should show them.
        restored2.grid_mut().scroll_display(Scroll::Top);

        // Verify row widths.
        let grid = restored2.grid();
        for line_idx in grid.topmost_line().0..grid.screen_lines() as i32 {
            let line = Line(line_idx);
            let row_len = grid[line].len();
            assert_eq!(
                row_len, 171,
                "row at line {line_idx} has {row_len} columns, expected 171"
            );
        }

        // Exercise crash paths.
        restored2.grid_mut().scroll_display(Scroll::Bottom);
        let mut parser: ansi::Processor = ansi::Processor::new();
        parser.advance(&mut restored2, b"\x1b[2K");
        parser.advance(&mut restored2, b"test input");
    }

    #[test]
    fn restore_scrolled_to_top_of_total_history() {
        // Scroll to absolute top (thawing all compressed rows), snapshot,
        // restore. After restore compressed rows are gone but display_offset
        // and hot history should both reflect the full thawed state.
        let cols = 80;
        let rows = 24;
        let mut term = term_with_scrollback(cols, rows, 200);

        term.grid_mut().compact_scrollback_if_needed();
        assert!(term.grid().compressed_history_len() > 0);

        // Scroll to absolute top — thaws everything.
        term.grid_mut().scroll_display_with_thaw(Scroll::Top);
        assert_eq!(term.grid().compressed_history_len(), 0);
        let offset = term.grid().display_offset();
        assert!(offset > 0);

        let mut restored = zed_restore(&term, cols, rows);

        // Everything was thawed before snapshot, so all rows are in the
        // ring buffer. display_offset should still be valid.
        assert!(
            restored.grid().display_offset() <= restored.grid().history_size(),
            "display_offset ({}) > history_size ({})",
            restored.grid().display_offset(),
            restored.grid().history_size(),
        );

        // Compact + iterate.
        restored.grid_mut().compact_scrollback_if_needed();
        let content = restored.renderable_content();
        let _count = content.display_iter.count();

        // Input after compact.
        let mut parser: ansi::Processor = ansi::Processor::new();
        parser.advance(&mut restored, b"hello");
    }


}
