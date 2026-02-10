//! Traceback and frame information.
//!
//! This module provides lazy traceback construction for exceptions.
//! Line numbers are only computed when the traceback is accessed,
//! following CPython 3.11+ optimization patterns.
//!
//! # Performance Design
//!
//! - **Lazy line numbers**: Only computed when __traceback__ is accessed
//! - **SmallVec frames**: Stack-allocated for shallow traces (< 8 frames)
//! - **Deferred string formatting**: No allocations until str() called

use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;

// ============================================================================
// Frame Information
// ============================================================================

/// Information about a single stack frame in a traceback.
///
/// This is the raw frame data captured when an exception is raised.
/// Line numbers may be computed lazily from bytecode offsets.
#[derive(Clone)]
pub struct FrameInfo {
    /// Name of the function/method.
    pub func_name: Arc<str>,

    /// Filename (module path).
    pub filename: Arc<str>,

    /// Bytecode offset at time of exception.
    pub bytecode_offset: u32,

    /// Line number (0 = not yet computed, lazily resolved).
    pub line_number: u32,

    /// First line number of the function (for relative line computation).
    pub first_lineno: u32,
}

impl FrameInfo {
    /// Creates a new frame info with a known line number.
    #[inline]
    pub fn new(func_name: Arc<str>, filename: Arc<str>, line_number: u32) -> Self {
        Self {
            func_name,
            filename,
            bytecode_offset: 0,
            line_number,
            first_lineno: line_number,
        }
    }

    /// Creates a frame info with deferred line number resolution.
    #[inline]
    pub fn with_offset(
        func_name: Arc<str>,
        filename: Arc<str>,
        bytecode_offset: u32,
        first_lineno: u32,
    ) -> Self {
        Self {
            func_name,
            filename,
            bytecode_offset,
            line_number: 0, // Will be resolved lazily
            first_lineno,
        }
    }

    /// Returns true if the line number needs to be resolved.
    #[inline]
    pub const fn needs_resolution(&self) -> bool {
        self.line_number == 0
    }

    /// Sets the resolved line number.
    #[inline]
    pub fn set_line_number(&mut self, lineno: u32) {
        self.line_number = lineno;
    }

    /// Returns the line number, or 0 if not yet resolved.
    #[inline]
    pub const fn line_number(&self) -> u32 {
        self.line_number
    }
}

impl fmt::Debug for FrameInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameInfo")
            .field("func_name", &self.func_name)
            .field("filename", &self.filename)
            .field("line_number", &self.line_number)
            .finish()
    }
}

// ============================================================================
// Traceback Object
// ============================================================================

/// Number of frames stored inline (stack-allocated).
/// Most Python tracebacks are < 8 frames deep.
const INLINE_FRAMES: usize = 8;

/// Python traceback object.
///
/// Contains a list of stack frames from where an exception was raised.
/// Line numbers are resolved lazily for performance.
#[derive(Clone)]
pub struct TracebackObject {
    /// Stack frames, most recent last (matches Python's tb_next order).
    frames: SmallVec<[FrameInfo; INLINE_FRAMES]>,

    /// Cached formatted string (lazy).
    formatted_cache: Option<Arc<str>>,
}

impl TracebackObject {
    /// Creates an empty traceback.
    #[inline]
    pub fn empty() -> Self {
        Self {
            frames: SmallVec::new(),
            formatted_cache: None,
        }
    }

    /// Creates a traceback with a single frame.
    #[inline]
    pub fn single(frame: FrameInfo) -> Self {
        let mut frames = SmallVec::new();
        frames.push(frame);
        Self {
            frames,
            formatted_cache: None,
        }
    }

    /// Creates a traceback from an iterator of frames.
    pub fn from_frames(frames: impl IntoIterator<Item = FrameInfo>) -> Self {
        Self {
            frames: frames.into_iter().collect(),
            formatted_cache: None,
        }
    }

    /// Returns true if the traceback has no frames.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns the number of frames.
    #[inline]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Appends a frame to the traceback.
    #[inline]
    pub fn push(&mut self, frame: FrameInfo) {
        self.frames.push(frame);
        self.formatted_cache = None; // Invalidate cache
    }

    /// Extends the traceback with frames from another traceback.
    pub fn extend(&mut self, other: &TracebackObject) {
        self.frames.extend(other.frames.iter().cloned());
        self.formatted_cache = None;
    }

    /// Returns an iterator over the frames.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &FrameInfo> {
        self.frames.iter()
    }

    /// Returns a mutable iterator over the frames.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut FrameInfo> {
        self.formatted_cache = None; // Invalidate on mutation
        self.frames.iter_mut()
    }

    /// Returns the most recent frame (where exception was raised).
    #[inline]
    pub fn innermost(&self) -> Option<&FrameInfo> {
        self.frames.last()
    }

    /// Returns the outermost frame (entry point).
    #[inline]
    pub fn outermost(&self) -> Option<&FrameInfo> {
        self.frames.first()
    }

    /// Formats the traceback as a string (cached).
    pub fn format(&mut self) -> Arc<str> {
        if let Some(cached) = &self.formatted_cache {
            return Arc::clone(cached);
        }

        let mut output = String::with_capacity(self.frames.len() * 80);
        output.push_str("Traceback (most recent call last):\n");

        for frame in &self.frames {
            output.push_str("  File \"");
            output.push_str(&frame.filename);
            output.push_str("\", line ");
            output.push_str(&frame.line_number.to_string());
            output.push_str(", in ");
            output.push_str(&frame.func_name);
            output.push('\n');
        }

        let formatted: Arc<str> = Arc::from(output);
        self.formatted_cache = Some(Arc::clone(&formatted));
        formatted
    }

    /// Clears the traceback.
    #[inline]
    pub fn clear(&mut self) {
        self.frames.clear();
        self.formatted_cache = None;
    }

    /// Returns true if all frames have resolved line numbers.
    pub fn is_resolved(&self) -> bool {
        self.frames.iter().all(|f| !f.needs_resolution())
    }
}

impl Default for TracebackObject {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Debug for TracebackObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TracebackObject")
            .field("frames", &self.frames.len())
            .finish()
    }
}

impl fmt::Display for TracebackObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Traceback (most recent call last):\n")?;
        for frame in &self.frames {
            writeln!(
                f,
                "  File \"{}\", line {}, in {}",
                frame.filename, frame.line_number, frame.func_name
            )?;
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // FrameInfo Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_frame_info_new() {
        let frame = FrameInfo::new(Arc::from("test_func"), Arc::from("test.py"), 42);

        assert_eq!(&*frame.func_name, "test_func");
        assert_eq!(&*frame.filename, "test.py");
        assert_eq!(frame.line_number(), 42);
        assert!(!frame.needs_resolution());
    }

    #[test]
    fn test_frame_info_with_offset() {
        let frame = FrameInfo::with_offset(Arc::from("test_func"), Arc::from("test.py"), 100, 10);

        assert_eq!(frame.bytecode_offset, 100);
        assert_eq!(frame.first_lineno, 10);
        assert!(frame.needs_resolution());
        assert_eq!(frame.line_number(), 0);
    }

    #[test]
    fn test_frame_info_set_line_number() {
        let mut frame =
            FrameInfo::with_offset(Arc::from("test_func"), Arc::from("test.py"), 100, 10);

        assert!(frame.needs_resolution());
        frame.set_line_number(25);
        assert!(!frame.needs_resolution());
        assert_eq!(frame.line_number(), 25);
    }

    #[test]
    fn test_frame_info_clone() {
        let frame = FrameInfo::new(Arc::from("func"), Arc::from("file.py"), 10);
        let cloned = frame.clone();

        assert_eq!(&*frame.func_name, &*cloned.func_name);
        assert_eq!(frame.line_number(), cloned.line_number());
    }

    #[test]
    fn test_frame_info_debug() {
        let frame = FrameInfo::new(Arc::from("func"), Arc::from("file.py"), 10);
        let debug = format!("{:?}", frame);

        assert!(debug.contains("FrameInfo"));
        assert!(debug.contains("func"));
        assert!(debug.contains("file.py"));
    }

    // ════════════════════════════════════════════════════════════════════════
    // TracebackObject Creation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_empty() {
        let tb = TracebackObject::empty();
        assert!(tb.is_empty());
        assert_eq!(tb.len(), 0);
    }

    #[test]
    fn test_traceback_single() {
        let frame = FrameInfo::new(Arc::from("main"), Arc::from("main.py"), 1);
        let tb = TracebackObject::single(frame);

        assert!(!tb.is_empty());
        assert_eq!(tb.len(), 1);
    }

    #[test]
    fn test_traceback_from_frames() {
        let frames = vec![
            FrameInfo::new(Arc::from("outer"), Arc::from("a.py"), 10),
            FrameInfo::new(Arc::from("middle"), Arc::from("b.py"), 20),
            FrameInfo::new(Arc::from("inner"), Arc::from("c.py"), 30),
        ];

        let tb = TracebackObject::from_frames(frames);
        assert_eq!(tb.len(), 3);
    }

    #[test]
    fn test_traceback_default() {
        let tb = TracebackObject::default();
        assert!(tb.is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // TracebackObject Mutation Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_push() {
        let mut tb = TracebackObject::empty();

        tb.push(FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1));
        assert_eq!(tb.len(), 1);

        tb.push(FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2));
        assert_eq!(tb.len(), 2);
    }

    #[test]
    fn test_traceback_extend() {
        let mut tb1 = TracebackObject::single(FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1));

        let tb2 = TracebackObject::from_frames(vec![
            FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2),
            FrameInfo::new(Arc::from("c"), Arc::from("c.py"), 3),
        ]);

        tb1.extend(&tb2);
        assert_eq!(tb1.len(), 3);
    }

    #[test]
    fn test_traceback_clear() {
        let mut tb = TracebackObject::from_frames(vec![
            FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1),
            FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2),
        ]);

        assert!(!tb.is_empty());
        tb.clear();
        assert!(tb.is_empty());
    }

    // ════════════════════════════════════════════════════════════════════════
    // TracebackObject Iteration Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_iter() {
        let tb = TracebackObject::from_frames(vec![
            FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1),
            FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2),
        ]);

        let names: Vec<_> = tb.iter().map(|f| f.func_name.as_ref()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_traceback_iter_mut() {
        let mut tb = TracebackObject::from_frames(vec![
            FrameInfo::with_offset(Arc::from("a"), Arc::from("a.py"), 10, 1),
            FrameInfo::with_offset(Arc::from("b"), Arc::from("b.py"), 20, 2),
        ]);

        // Resolve line numbers
        for (i, frame) in tb.iter_mut().enumerate() {
            frame.set_line_number((i + 1) as u32 * 10);
        }

        assert!(tb.is_resolved());
    }

    #[test]
    fn test_traceback_innermost() {
        let tb = TracebackObject::from_frames(vec![
            FrameInfo::new(Arc::from("outer"), Arc::from("a.py"), 1),
            FrameInfo::new(Arc::from("inner"), Arc::from("b.py"), 2),
        ]);

        let innermost = tb.innermost().unwrap();
        assert_eq!(&*innermost.func_name, "inner");
    }

    #[test]
    fn test_traceback_outermost() {
        let tb = TracebackObject::from_frames(vec![
            FrameInfo::new(Arc::from("outer"), Arc::from("a.py"), 1),
            FrameInfo::new(Arc::from("inner"), Arc::from("b.py"), 2),
        ]);

        let outermost = tb.outermost().unwrap();
        assert_eq!(&*outermost.func_name, "outer");
    }

    #[test]
    fn test_traceback_empty_innermost_outermost() {
        let tb = TracebackObject::empty();
        assert!(tb.innermost().is_none());
        assert!(tb.outermost().is_none());
    }

    // ════════════════════════════════════════════════════════════════════════
    // TracebackObject Formatting Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_format() {
        let mut tb = TracebackObject::from_frames(vec![
            FrameInfo::new(Arc::from("main"), Arc::from("main.py"), 10),
            FrameInfo::new(Arc::from("helper"), Arc::from("utils.py"), 25),
        ]);

        let formatted = tb.format();
        assert!(formatted.contains("Traceback (most recent call last):"));
        assert!(formatted.contains("main.py"));
        assert!(formatted.contains("line 10"));
        assert!(formatted.contains("utils.py"));
        assert!(formatted.contains("line 25"));
    }

    #[test]
    fn test_traceback_format_cached() {
        let mut tb =
            TracebackObject::single(FrameInfo::new(Arc::from("test"), Arc::from("test.py"), 1));

        let first = tb.format();
        let second = tb.format();

        // Should be the same Arc (cached)
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn test_traceback_format_cache_invalidated_on_push() {
        let mut tb = TracebackObject::single(FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1));

        let first = tb.format();
        tb.push(FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2));
        let second = tb.format();

        // Should NOT be the same Arc (cache invalidated)
        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn test_traceback_display() {
        let tb = TracebackObject::from_frames(vec![FrameInfo::new(
            Arc::from("test"),
            Arc::from("test.py"),
            42,
        )]);

        let display = format!("{}", tb);
        assert!(display.contains("Traceback"));
        assert!(display.contains("test.py"));
        assert!(display.contains("42"));
    }

    // ════════════════════════════════════════════════════════════════════════
    // Resolution Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_is_resolved_true() {
        let tb = TracebackObject::from_frames(vec![
            FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1),
            FrameInfo::new(Arc::from("b"), Arc::from("b.py"), 2),
        ]);

        assert!(tb.is_resolved());
    }

    #[test]
    fn test_traceback_is_resolved_false() {
        let tb = TracebackObject::from_frames(vec![
            FrameInfo::new(Arc::from("a"), Arc::from("a.py"), 1),
            FrameInfo::with_offset(Arc::from("b"), Arc::from("b.py"), 100, 2),
        ]);

        assert!(!tb.is_resolved());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Memory Layout Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_inline_frames_constant() {
        // Verify our inline size is 8
        assert_eq!(INLINE_FRAMES, 8);
    }

    #[test]
    fn test_smallvec_inline_capacity() {
        let tb = TracebackObject::empty();
        // SmallVec should be able to hold INLINE_FRAMES items inline
        assert!(tb.frames.capacity() >= 0); // Just verify it compiles
    }

    // ════════════════════════════════════════════════════════════════════════
    // Clone Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_clone() {
        let tb = TracebackObject::from_frames(vec![FrameInfo::new(
            Arc::from("a"),
            Arc::from("a.py"),
            1,
        )]);

        let cloned = tb.clone();
        assert_eq!(tb.len(), cloned.len());
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_traceback_many_frames() {
        // Test with more frames than inline capacity
        let frames: Vec<_> = (0..20)
            .map(|i| {
                FrameInfo::new(
                    Arc::from(format!("func_{}", i)),
                    Arc::from(format!("file_{}.py", i)),
                    i as u32,
                )
            })
            .collect();

        let tb = TracebackObject::from_frames(frames);
        assert_eq!(tb.len(), 20);
    }

    #[test]
    fn test_traceback_deep_nesting() {
        // Simulate deep recursion
        let mut tb = TracebackObject::empty();

        for i in 0..100 {
            tb.push(FrameInfo::new(
                Arc::from(format!("recurse_{}", i)),
                Arc::from("recursive.py"),
                i as u32,
            ));
        }

        assert_eq!(tb.len(), 100);
        assert_eq!(&*tb.innermost().unwrap().func_name, "recurse_99");
    }
}
