//! Source span tracking for error reporting.
//!
//! Spans represent byte offset ranges in source files, enabling precise
//! error messages and source mapping for debugging.

use std::fmt;
use std::ops::Range;

/// A span representing a byte offset range in source code.
///
/// Spans are half-open intervals: `[start, end)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    /// Start byte offset (inclusive).
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
}

impl Span {
    /// Create a new span from start to end.
    #[inline]
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Create an empty span at a position.
    #[inline]
    #[must_use]
    pub const fn empty(pos: u32) -> Self {
        Self {
            start: pos,
            end: pos,
        }
    }

    /// Create a span covering a single byte.
    #[inline]
    #[must_use]
    pub const fn single(pos: u32) -> Self {
        Self {
            start: pos,
            end: pos + 1,
        }
    }

    /// Create a dummy span for generated code.
    #[inline]
    #[must_use]
    pub const fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }

    /// Check if this span is a dummy span.
    #[inline]
    #[must_use]
    pub const fn is_dummy(&self) -> bool {
        self.start == 0 && self.end == 0
    }

    /// Get the length of this span in bytes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// Check if the span is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Check if this span contains a byte offset.
    #[inline]
    #[must_use]
    pub const fn contains(&self, offset: u32) -> bool {
        offset >= self.start && offset < self.end
    }

    /// Check if this span fully contains another span.
    #[inline]
    #[must_use]
    pub const fn contains_span(&self, other: Span) -> bool {
        self.start <= other.start && self.end >= other.end
    }

    /// Check if this span overlaps with another.
    #[inline]
    #[must_use]
    pub const fn overlaps(&self, other: Span) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Merge two spans into one covering both.
    #[inline]
    #[must_use]
    pub const fn merge(self, other: Span) -> Span {
        let start = if self.start < other.start {
            self.start
        } else {
            other.start
        };
        let end = if self.end > other.end {
            self.end
        } else {
            other.end
        };
        Span { start, end }
    }

    /// Extend this span to include another.
    #[inline]
    pub fn extend(&mut self, other: Span) {
        if other.start < self.start {
            self.start = other.start;
        }
        if other.end > self.end {
            self.end = other.end;
        }
    }

    /// Shrink the span by removing bytes from start and end.
    #[inline]
    #[must_use]
    pub const fn shrink(self, start_bytes: u32, end_bytes: u32) -> Span {
        let new_start = self.start.saturating_add(start_bytes);
        let new_end = self.end.saturating_sub(end_bytes);
        if new_start >= new_end {
            Span::empty(self.start)
        } else {
            Span::new(new_start, new_end)
        }
    }

    /// Get the byte range as a `Range<usize>`.
    #[inline]
    #[must_use]
    pub const fn as_range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }

    /// Get a subslice of source text covered by this span.
    #[inline]
    #[must_use]
    pub fn slice<'a>(&self, source: &'a str) -> &'a str {
        let start = self.start as usize;
        let end = self.end as usize;
        if start <= end && end <= source.len() {
            &source[start..end]
        } else {
            ""
        }
    }

    /// Get the (1-indexed) line and column from source.
    #[must_use]
    pub fn line_col(&self, source: &str) -> (usize, usize) {
        let offset = self.start as usize;
        if offset > source.len() {
            return (1, 1);
        }

        let prefix = &source[..offset];
        let line = prefix.chars().filter(|&c| c == '\n').count() + 1;
        let last_newline = prefix.rfind('\n').map_or(0, |pos| pos + 1);
        let col = offset - last_newline + 1;

        (line, col)
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

impl From<Range<u32>> for Span {
    fn from(range: Range<u32>) -> Self {
        Self::new(range.start, range.end)
    }
}

impl From<Range<usize>> for Span {
    fn from(range: Range<usize>) -> Self {
        Self::new(range.start as u32, range.end as u32)
    }
}

impl From<Span> for Range<usize> {
    fn from(span: Span) -> Self {
        span.as_range()
    }
}

/// A value with an associated source span.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    /// The value.
    pub value: T,
    /// The source span.
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Create a new spanned value.
    #[inline]
    #[must_use]
    pub const fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    /// Map the inner value with a function.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U> {
        Spanned {
            value: f(self.value),
            span: self.span,
        }
    }

    /// Get a reference to the inner value.
    #[inline]
    #[must_use]
    pub const fn as_ref(&self) -> Spanned<&T> {
        Spanned {
            value: &self.value,
            span: self.span,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} @ {:?}", self.value, self.span)
    }
}

impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_new() {
        let span = Span::new(10, 20);
        assert_eq!(span.start, 10);
        assert_eq!(span.end, 20);
    }

    #[test]
    fn test_span_empty() {
        let span = Span::empty(15);
        assert_eq!(span.start, 15);
        assert_eq!(span.end, 15);
        assert!(span.is_empty());
    }

    #[test]
    fn test_span_single() {
        let span = Span::single(5);
        assert_eq!(span.start, 5);
        assert_eq!(span.end, 6);
        assert_eq!(span.len(), 1);
        assert!(!span.is_empty());
    }

    #[test]
    fn test_span_dummy() {
        let span = Span::dummy();
        assert!(span.is_dummy());
        assert!(span.is_empty());
    }

    #[test]
    fn test_span_len() {
        assert_eq!(Span::new(0, 10).len(), 10);
        assert_eq!(Span::new(5, 5).len(), 0);
        assert_eq!(Span::new(100, 200).len(), 100);
    }

    #[test]
    fn test_span_is_empty() {
        assert!(Span::new(5, 5).is_empty());
        assert!(Span::new(10, 5).is_empty()); // Invalid span
        assert!(!Span::new(0, 1).is_empty());
    }

    #[test]
    fn test_span_contains_offset() {
        let span = Span::new(10, 20);
        assert!(!span.contains(9));
        assert!(span.contains(10));
        assert!(span.contains(15));
        assert!(span.contains(19));
        assert!(!span.contains(20));
    }

    #[test]
    fn test_span_contains_span() {
        let outer = Span::new(10, 50);
        let inner = Span::new(20, 30);
        let overlapping = Span::new(5, 25);
        let outside = Span::new(60, 70);

        assert!(outer.contains_span(inner));
        assert!(outer.contains_span(outer)); // Contains itself
        assert!(!outer.contains_span(overlapping));
        assert!(!outer.contains_span(outside));
    }

    #[test]
    fn test_span_overlaps() {
        let span1 = Span::new(10, 20);
        let span2 = Span::new(15, 25);
        let span3 = Span::new(20, 30);
        let span4 = Span::new(0, 5);

        assert!(span1.overlaps(span2));
        assert!(span2.overlaps(span1));
        assert!(!span1.overlaps(span3)); // Adjacent, not overlapping
        assert!(!span1.overlaps(span4));
    }

    #[test]
    fn test_span_merge() {
        let span1 = Span::new(10, 20);
        let span2 = Span::new(15, 30);
        let merged = span1.merge(span2);

        assert_eq!(merged.start, 10);
        assert_eq!(merged.end, 30);
    }

    #[test]
    fn test_span_merge_non_overlapping() {
        let span1 = Span::new(0, 10);
        let span2 = Span::new(20, 30);
        let merged = span1.merge(span2);

        assert_eq!(merged.start, 0);
        assert_eq!(merged.end, 30);
    }

    #[test]
    fn test_span_extend() {
        let mut span = Span::new(10, 20);
        span.extend(Span::new(5, 15));
        assert_eq!(span.start, 5);
        assert_eq!(span.end, 20);

        span.extend(Span::new(15, 30));
        assert_eq!(span.start, 5);
        assert_eq!(span.end, 30);
    }

    #[test]
    fn test_span_shrink() {
        let span = Span::new(10, 30);
        let shrunk = span.shrink(5, 5);
        assert_eq!(shrunk.start, 15);
        assert_eq!(shrunk.end, 25);
    }

    #[test]
    fn test_span_shrink_to_empty() {
        let span = Span::new(10, 20);
        let shrunk = span.shrink(10, 10);
        assert!(shrunk.is_empty());
    }

    #[test]
    fn test_span_shrink_overflow() {
        let span = Span::new(10, 15);
        let shrunk = span.shrink(10, 10);
        assert!(shrunk.is_empty());
    }

    #[test]
    fn test_span_as_range() {
        let span = Span::new(5, 15);
        let range = span.as_range();
        assert_eq!(range, 5..15);
    }

    #[test]
    fn test_span_slice() {
        let source = "hello world";
        let span = Span::new(0, 5);
        assert_eq!(span.slice(source), "hello");

        let span2 = Span::new(6, 11);
        assert_eq!(span2.slice(source), "world");
    }

    #[test]
    fn test_span_slice_out_of_bounds() {
        let source = "short";
        let span = Span::new(0, 100);
        assert_eq!(span.slice(source), "");
    }

    #[test]
    fn test_span_line_col_first_line() {
        let source = "hello world";
        let span = Span::new(0, 5);
        assert_eq!(span.line_col(source), (1, 1));

        let span2 = Span::new(6, 11);
        assert_eq!(span2.line_col(source), (1, 7));
    }

    #[test]
    fn test_span_line_col_multiline() {
        let source = "line1\nline2\nline3";

        // Start of line 1
        assert_eq!(Span::new(0, 1).line_col(source), (1, 1));

        // Start of line 2
        assert_eq!(Span::new(6, 7).line_col(source), (2, 1));

        // Middle of line 2
        assert_eq!(Span::new(8, 9).line_col(source), (2, 3));

        // Start of line 3
        assert_eq!(Span::new(12, 13).line_col(source), (3, 1));
    }

    #[test]
    fn test_span_line_col_at_newline() {
        let source = "abc\ndef";
        let span = Span::new(3, 4); // At the newline
        assert_eq!(span.line_col(source), (1, 4));
    }

    #[test]
    fn test_span_debug() {
        let span = Span::new(10, 20);
        assert_eq!(format!("{:?}", span), "10..20");
    }

    #[test]
    fn test_span_display() {
        let span = Span::new(10, 20);
        assert_eq!(format!("{}", span), "[10, 20)");
    }

    #[test]
    fn test_span_from_range_u32() {
        let span: Span = (5u32..15u32).into();
        assert_eq!(span.start, 5);
        assert_eq!(span.end, 15);
    }

    #[test]
    fn test_span_from_range_usize() {
        let span: Span = (5usize..15usize).into();
        assert_eq!(span.start, 5);
        assert_eq!(span.end, 15);
    }

    #[test]
    fn test_span_into_range() {
        let span = Span::new(5, 15);
        let range: Range<usize> = span.into();
        assert_eq!(range, 5..15);
    }

    #[test]
    fn test_span_equality() {
        let span1 = Span::new(10, 20);
        let span2 = Span::new(10, 20);
        let span3 = Span::new(10, 21);

        assert_eq!(span1, span2);
        assert_ne!(span1, span3);
    }

    #[test]
    fn test_span_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Span::new(0, 10));
        set.insert(Span::new(0, 10)); // Duplicate
        set.insert(Span::new(5, 15));

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_span_default() {
        let span = Span::default();
        assert_eq!(span.start, 0);
        assert_eq!(span.end, 0);
        assert!(span.is_dummy());
    }

    #[test]
    fn test_span_clone_copy() {
        let span = Span::new(10, 20);
        let copied = span;
        let cloned = span.clone();

        assert_eq!(span, copied);
        assert_eq!(span, cloned);
    }

    // Spanned tests

    #[test]
    fn test_spanned_new() {
        let spanned = Spanned::new(42, Span::new(0, 2));
        assert_eq!(spanned.value, 42);
        assert_eq!(spanned.span, Span::new(0, 2));
    }

    #[test]
    fn test_spanned_map() {
        let spanned = Spanned::new(21, Span::new(0, 2));
        let doubled = spanned.map(|x| x * 2);

        assert_eq!(doubled.value, 42);
        assert_eq!(doubled.span, Span::new(0, 2));
    }

    #[test]
    fn test_spanned_as_ref() {
        let spanned = Spanned::new(String::from("hello"), Span::new(0, 5));
        let ref_spanned = spanned.as_ref();

        assert_eq!(ref_spanned.value, &String::from("hello"));
        assert_eq!(ref_spanned.span, Span::new(0, 5));
    }

    #[test]
    fn test_spanned_debug() {
        let spanned = Spanned::new("test", Span::new(0, 4));
        let debug = format!("{:?}", spanned);
        assert!(debug.contains("test"));
        assert!(debug.contains("0..4"));
    }

    #[test]
    fn test_spanned_display() {
        let spanned = Spanned::new(42, Span::new(0, 2));
        assert_eq!(format!("{}", spanned), "42");
    }

    #[test]
    fn test_spanned_equality() {
        let s1 = Spanned::new(42, Span::new(0, 2));
        let s2 = Spanned::new(42, Span::new(0, 2));
        let s3 = Spanned::new(42, Span::new(0, 3)); // Different span
        let s4 = Spanned::new(43, Span::new(0, 2)); // Different value

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s1, s4);
    }

    #[test]
    fn test_spanned_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(Spanned::new("a", Span::new(0, 1)));
        set.insert(Spanned::new("a", Span::new(0, 1))); // Duplicate
        set.insert(Spanned::new("b", Span::new(0, 1)));

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_spanned_clone() {
        let original = Spanned::new(42, Span::new(0, 2));
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }
}
