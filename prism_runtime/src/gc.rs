//! Garbage collector (placeholder).
/// GC state.
pub struct Gc;
impl Gc {
    /// Create new GC.
    pub fn new() -> Self {
        Self
    }
}
impl Default for Gc {
    fn default() -> Self {
        Self::new()
    }
}
