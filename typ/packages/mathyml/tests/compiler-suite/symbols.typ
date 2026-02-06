// Test math symbol edge cases.

--- math-symbol-basic ---
#let sym = symbol("s", ("basic", "s"))
#assert.eq($sym.basic$, $s$)

--- math-symbol-double ---
#let sym = symbol("s", ("test.basic", "s"))
#assert.eq($sym.test.basic$, $s$)
