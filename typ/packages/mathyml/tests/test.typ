#import "../src/lib.typ" as mathyml: to-mathml
#import mathyml.prelude: *

#context if mathyml.is-html() {
html.elem("style")[
  #```CSS
  :root {
    font-family: Libertinus Serif;
    margin: 3em;
    font-size: 16pt;
  }
  h2, h3 {
    text-decoration: underline;
  }
  ```.text
]}

#mathyml.stylesheets()

#set text(size: 17pt)

The fraction #to-mathml($1/3$) is not a decimal number.
#to-mathml($ 1/3 $)

#let test(eq, block: true) = {
  let test-counter = counter("tests")
  test-counter.step()
  [Test #context test-counter.display():]
  if not block {
    html.elem("span", attrs: (style: "text-align: center; display: inline-block; width: 90%"), to-mathml(block: block, eq))
    linebreak()
  } else {
    to-mathml(block: block, eq)
  }
}

#test($pi(x)$)
#test($|x| = x "iff" x >= 0$)
#test($sin x$)
#test($Gamma "Γ"$)
#test($+i$) // prefix plus
#test($sum i$)
#test($forall A in op("sl")(n, FF), op("Tr") A = 0$)
#test($1/2$)
#test($sqrt(42)$)
#test($root(5, 42)$)
#test($sqrt(2)/root(3, 4) + root(6/7, 5) + sqrt(8-9)$)
#test($root(3, sqrt(1/2))$)
#test($binom(3, 2) = 3 != 3/2$)
#test($binom(3, 2, 3, 4)$)
#test($x_1$)
#test($x^2$)
#test($x^2_1$)
#test($attach(x, bl: 3, tl: 4)$)
#test($attach(x, tr: 1, br: 2, bl: 3, tl: 4)$)
#test($attach(x, tr: 1, tl: 4)$)
#test($attach(x, br: 2, bl: 3)$)
#test($sum_(i=1)^n a_i$)

#test($vec(a, b, c)$)
#test($vec(align: #right, -1, 1, -1)$)
#test($vec(gap: #1em, a, b, c)$)
#test($vec(gap: #(1em + 5pt), a, b, c)$)
#test($vec(gap: #50%, a, b, c)$)
#test($vec(delim: "[", a, b, c)$)
#test($vec(delim: #("[", ")"), a, b, c)$)
#test($mat(
  1, 2, ..., 10;
  2, 2, ..., 10;
  dots.v, dots.v, dots.down, dots.v;
  10, 10, ..., 10;
)$)

#test($f(x) = cases(
  -1 &"if" x < 0,
  1 &"else"
)$)
#test($cases(reverse: #true, gap: #1em,
  -1 &"if" x < 0,
  1 &"else"
) = f(x)$)
#test($f(x) = cases(delim: "(",
  -1 &"if" x < 0,
  1 &"else"
)$)

#test($underline(123) "" underbrace(123) "" underbracket(123) "" underparen(123) "" undershell(123)$)
#test($overline(123) "" overbrace(123) "" overbracket(123) "" overparen(123) "" overshell(123)$)
#test($underbrace(123, 456) "" underbracket(123, 456) "" underparen(123, 456) "" undershell(123, 456)$)
#test($overbrace(123, 456) "" overbracket(123, 456) "" overparen(123, 456) "" overshell(123, 456)$)

#test($grave(a) = accent(a, `)$)
#test($arrow(a) = accent(a, arrow) and tilde(a) = accent(a, \u{0303})$)

#test($a'''_b = a^'''_b$)
#test($a^primes(#0) = a' = a'' = a''' = a'''' = a^primes(#5) = a^primes(#6)$)

#test({
  let loves = math.class("relation",sym.suit.heart)
  $x loves y and y loves 5$
})
#test($class("unary", x) b class("punctuation", t) a$)

#test($H stretch(=)^"define" U + p V$)
#test($tan x = (sin x)/(cos x)$)
#test($op("custom")_(n->oo) n$)

#test($"ord"(G)$)
#test($class("unary", "Gal")(L slash K)$)

#test($frak(A)$)
#test($sans(A B C)$)
#test($frak(P)$)
#test($mono(x + y = z)$)
#test($bb(b)$)
#test($bb(N) = NN$)
#test($f: NN -> RR$)
#test($cal(P)$)

#test($sum_i x_i/2 = display(sum_i x_i/2)$, block: false)
#test($ sum_i x_i/2
    = inline(sum_i x_i/2) $)
#test($sum_i x_i/2 = script(sum_i x_i/2)$)
#test($sum_i x_i/2 = sscript(sum_i x_i/2)$)

#test($
  a &= 1/2 \
  sum_i &= 3
$)

#test($ (3x + y) / 7 &= 9 && "given" \
  3x + y &= 63 & "multiply by 7" \
  3x &= 63 - y && "subtract y" \
  x &= 21 - y/3 & "divide by 3" $)

#test($tilde(sum)$)

#test($ √2^3 = sqrt(2^3) $)

#test($ "abc" := bold("abc") := italic("abc") := upright("abc") $)
#test($ A := bold(A) := italic(A) := upright(A) $)
#test($ Gamma := bold(Gamma) := italic(Gamma) := upright(Gamma) $)

#test($integral x dif x$)
#test($integral x Dif x$)

#test($cases(
  1 quad &"if" (x dot y)/2 <= 0,
  2 &"if" x divides 2,
  3 &"if" x in NN,
  4 &"else",
)$)

#test($
  sqrt(f)/f
  sqrt(f^2)/f^2
  sqrt(f'^2)/f'^2
  sqrt(f''_n^2)/f''^2_n
$)

#[
  #let a0 = math.attach(math.alpha, b: [0])
  #let a1 = $alpha^1$
  #let a2 = $attach(a1, bl: 3)$
  #test($ a0 + a1 + a0_2 \
    a1_2 + a0^2 + a1^2 \
    a2 + a2_2 + a2^2 $)
]

#test($ lr(a/b\]) = a = lr(\{a/b) $)

#test($ mat(1, 0, 1; 0, 1, 2; augment: #2) $)
#test($ mat(augment: #2, 1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 1) $)
#test($mat(augment: #(vline: (-2, 2, 1), hline: (1, -1), stroke: 0.1em),
  11, 12, 13, 14, 15;
  21, 22, 23, 24, 25;
  31, 32, 33, 34, 35;
  41, 42, 43, 44, 45;
)$)
#test($ mat(augment: #2, 1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 1) $)
#test($ mat(1, 0, 1; 0, 1, 2; augment: #(-1)) $)
#test($ mat(0, 0, 0; 1, 1, 1; augment: #(hline: 1, stroke: 2pt)) $)

#test($ { x mid(|) sum_(i=1)^n | f_i (x) < 1 } $)

#test($lr([a], size: #200%)$)

#test($
  a &= c + 1 & "Some longer text" \
  &= d + 100 \
  &= x & & 1
$)

#[
  #show math.equation.where(block: true): to-mathml.with(block: true)
  #show math.equation.where(block: false): to-mathml.with(block: false)

  To solve the cubic equation $t^3 + p t + q = 0$ (where the real numbers
  $p, q$ satisfy $4p^3 + 27q^2 > 0$) one can use Cardano's formula:
  $
    root(3, -q/2 + sqrt(q^2/4 + p^3/27))
    + root(3, -q/2 - sqrt(q^2/4 + p^3/27)).
  $
  For any $u_1, ..., u_n in CC$ and $v_1, ..., v_n in CC$, the Cauchy-Bunyakovsky-Schwarz inequality can be written as follows:
  $
    abs(sum_(k=1)^n u_k overline(v_k))^2 <= (sum_(k=1)^n |u_k|)^2 (sum_(k=1)^n |v_k|)^2. \
  $
  Finally, the determinant of a Vandermonde matrix can be calculated using the following expression:
  $
    mat(delim: "|",
      1, x_1, x_1^2, ..., x_1^(n-1);
      1, x_2, x_2^2, ..., x_2^(n-1);
      1, x_3, x_3^2, ..., x_3^(n-1);
      dots.v, dots.v, dots.v, dots.down, dots.v;
      1, x_n, x_n^2, ..., x_n^(n-1);
    ) = product_(1 <= i, j <= n) (x_i - x_j)
  $
]
