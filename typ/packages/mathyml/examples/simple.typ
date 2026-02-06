#import "../src/lib.typ" as mathyml: to-mathml
// include the prelude for `bold`, `frak` etc.
#import mathyml.prelude: *

// include simple styling
#context if mathyml.is-html() {
html.elem("style")[
  #```CSS
  :root {
    font-family: Libertinus Serif;
    margin: 3em;
    font-size: 16pt;
  }
  ```.text
]}

// include the required stylesheet (and the mathfont)
#mathyml.stylesheets()

#set text(size: 17pt)

// convert equations manually, but only for html
The fraction #to-mathml($1/3$) is not a decimal number. And we know
#to-mathml($ a^2 + b^2 = c^2. $)


// convert equations automatically, but only if we export to html.
// If this panics, try `try-to-mathml` instead
#show math.equation: to-mathml

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
