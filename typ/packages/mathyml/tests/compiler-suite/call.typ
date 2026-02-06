// Test math function call edge cases.

// Note: 2d argument calls are tested for matrices in `mat.typ`

--- math-call-non-func ---
$ pi(a) $
$ pi(a,) $
$ pi(a,b) $
$ pi(a,b,) $


--- math-call-named-args ---
#let func1(my: none) = my
#let func2(_my: none) = _my
#let func3(my-body: none) = my-body
#let func4(_my-body: none) = _my-body
#let func5(m: none) = m
$ func1(my: a) $
$ func2(_my: a) $
$ func3(my-body: a) $
$ func4(_my-body: a) $
$ func5(m: a) $
$ func5(m: sigma : f) $
$ func5(m: sigma:pi) $


--- math-call-spread-shorthand-clash ---
#let func(body) = body
$func(...)$

#let test-repr(it, r) = [#repr(it) #r]

--- math-call-spread-repr ---
#let args(..body) = body
#let check(it, r) = test-repr(it.body.text, r)
#check($args(..#range(0, 4).chunks(2))$, "arguments((0, 1), (2, 3))")
#check($#args(range(1, 5).chunks(2))$, "arguments(((1, 2), (3, 4)))")
#check($#args(..range(1, 5).chunks(2))$, "arguments((1, 2), (3, 4))")
#check($args(#(..range(2, 6).chunks(2)))$, "arguments(((2, 3), (4, 5)))")
#let nums = range(0, 4).chunks(2)
#check($args(..nums)$, "arguments((0, 1), (2, 3))")
#check($args(..nums;)$, "arguments(((0, 1), (2, 3)))")
#check($args(..nums, ..nums)$, "arguments((0, 1), (2, 3), (0, 1), (2, 3))")
#check($args(..nums, 4, 5)$, "arguments((0, 1), (2, 3), [4], [5])")
#check($args(..nums, ..#range(4, 6))$, "arguments((0, 1), (2, 3), 4, 5)")
#check($args(..nums, #range(4, 6))$, "arguments((0, 1), (2, 3), (4, 5))")
#check($args(..nums, 1, 2; 3, 4)$, "arguments(((0, 1), (2, 3), [1], [2]), ([3], [4]))")
#check($args(1, 2; ..nums)$, "arguments(([1], [2]), ((0, 1), (2, 3)))")
#check($args(1, 2; 3, 4)$, "arguments(([1], [2]), ([3], [4]))")
#check($args(1, 2; 3, 4; ..#range(5, 7))$, "arguments(([1], [2]), ([3], [4]), (5, 6))")
#check($args(1, 2; 3, 4, ..#range(5, 7))$, "arguments(([1], [2]), ([3], [4], 5, 6))")
#check($args(1, 2; 3, 4, ..#range(5, 7);)$, "arguments(([1], [2]), ([3], [4], 5, 6))")
#check($args(1, 2; 3, 4, ..#range(5, 7),)$, "arguments(([1], [2]), ([3], [4], 5, 6))")

--- math-call-repr ---
#let args(..body) = body
#let check(it, r) = test-repr(it.body.text, r)
#check($args(a)$, "arguments([a])")
#check($args(a,)$, "arguments([a])")
#check($args(a,b)$, "arguments([a], [b])")
#check($args(a,b,)$, "arguments([a], [b])")
#check($args(,a,b,,,)$, "arguments([], [a], [b], [], [])")

--- math-call-2d-semicolon-priority ---
// If the semicolon directly follows a hash expression, it terminates that
// instead of indicating 2d arguments.
$ mat(#"math" ; "wins") $
$ mat(#"code"; "wins") $

--- math-call-2d-repr ---
#let args(..body) = body
#let check(it, r) = test-repr(it.body.text, r)
#check($args(a;b)$, "arguments(([a],), ([b],))")
#check($args(a,b;c)$, "arguments(([a], [b]), ([c],))")
#check($args(a,b;c,d;e,f)$, "arguments(([a], [b]), ([c], [d]), ([e], [f]))")

--- math-call-2d-named-repr ---
#let args(..body) = (body.pos(), body.named())
#let check(it, r) = test-repr(it.body.text, r)
#check($args(a: b)$, "((), (a: [b]))")
#check($args(1, 2; 3, 4)$, "((([1], [2]), ([3], [4])), (:))")
#check($args(a: b, 1, 2; 3, 4)$, "((([1], [2]), ([3], [4])), (a: [b]))")
#check($args(1, a: b, 2; 3, 4)$, "(([1], ([2],), ([3], [4])), (a: [b]))")
#check($args(1, 2, a: b; 3, 4)$, "(([1], [2], (), ([3], [4])), (a: [b]))")
#check($args(1, 2; a: b, 3, 4)$, "((([1], [2]), ([3], [4])), (a: [b]))")
#check($args(1, 2; 3, a: b, 4)$, "((([1], [2]), [3], ([4],)), (a: [b]))")
#check($args(1, 2; 3, 4, a: b)$, "((([1], [2]), [3], [4]), (a: [b]))")
#check($args(a: b, 1, 2, 3, c: d)$, "(([1], [2], [3]), (a: [b], c: [d]))")
#check($args(1, 2, 3; a: b)$, "((([1], [2], [3]),), (a: [b]))")
#check($args(a-b: a,, e:f;; d)$, "(([], (), ([],), ([d],)), (a-b: [a], e: [f]))")
#check($args(a: b, ..#range(0, 4))$, "((0, 1, 2, 3), (a: [b]))")

--- math-call-2d-escape-repr ---
#let args(..body) = body
#let check(it, r) = test-repr(it.body.text, r)
#check($args(a\;b)$, "arguments(sequence([a], [;], [b]))")
#check($args(a\,b;c)$, "arguments((sequence([a], [,], [b]),), ([c],))")
#check($args(b\;c\,d;e)$, "arguments((sequence([b], [;], [c], [,], [d]),), ([e],))")
#check($args(a\: b)$, "arguments(sequence([a], [:], [ ], [b]))")
#check($args(a : b)$, "arguments(sequence([a], [ ], [:], [ ], [b]))")
#check($args(\..a)$, "arguments(sequence([.], [.], [a]))")
#check($args(.. a)$, "arguments(sequence([.], [.], [ ], [a]))")
#check($args(a..b)$, "arguments(sequence([a], [.], [.], [b]))")

--- math-call-2d-repr-structure ---
#let args(..body) = body
#let check(it, r) = test-repr(it.body.text, r)
#check($args( a; b; )$, "arguments(([a],), ([b],))")
#check($args(a;  ; c)$, "arguments(([a],), ([],), ([c],))")
#check($args(a b,/**/; b)$, "arguments((sequence([a], [ ], [b]), []), ([b],))")
#check($args(a/**/b, ; b)$, "arguments((sequence([a], [b]), []), ([b],))")
#check($args( ;/**/a/**/b/**/; )$, "arguments(([],), (sequence([a], [b]),))")
#check($args( ; , ; )$, "arguments(([],), ([], []))")
#check($args(/**/; // funky whitespace/trivia
    ,   /**/  ;/**/)$, "arguments(([],), ([], []))")

--- math-call-empty-args-non-func ---
// Trailing commas and empty args introduce blank content in math
$ sin(,x,y,,,) $
// with whitespace/trivia:
$ sin( ,/**/x/**/, , /**/y, ,/**/, ) $

--- math-call-empty-args-repr ---
#let args(..body) = body
#let check(it, r) = test-repr(it.body.text, r)
#check($args(,x,,y,,)$, "arguments([], [x], [], [y], [])")
// with whitespace/trivia:
#check($args( ,/**/x/**/, , /**/y, ,/**/, )$, "arguments([], [x], [], [y], [], [])")

--- math-call-value-non-func ---
$ sin(1) $

--- math-call-pass-to-box ---
// When passing to a function, we lose the italic styling if we wrap the content
// in a non-math function unless it's already nested in some math element (lr,
// attach, etc.)
//
// This is not good, so this test should fail and be updated once it is fixed.
#let id(body) = body
#let bx(body) = box(body, stroke: blue+0.5pt, inset: (x:2pt, y:3pt))
#let eq(body) = math.equation(body)
$
     x y   &&quad     x (y z)   &quad     x y^z  \
  id(x y)  &&quad  id(x (y z))  &quad  id(x y^z) \
  eq(x y)  &&quad  eq(x (y z))  &quad  eq(x y^z) \
  bx(x y)  &&quad  bx(x (y z))  &quad  bx(x y^z) \
$

--- issue-3774-math-call-empty-2d-args ---
$ mat(;,) $
// Add some whitespace/trivia:
$ mat(; ,) $
$ mat(;/**/,) $
$ mat(;
,) $
$ mat(;// line comment
,) $
$ mat(
  1, , ;
   ,1, ;
   , ,1;
) $
