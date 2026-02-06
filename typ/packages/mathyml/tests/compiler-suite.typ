#import "../src/lib.typ" as mathyml: to-mathml

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

#show math.equation: mathyml.maybe-html.with(to-mathml)

#show regex("—.*—"): it => it + linebreak()

// currently: 54 tests disabled of 180

#include "compiler-suite/accent.typ"
#include "compiler-suite/alignment.typ"
#include "compiler-suite/attach.typ"
#include "compiler-suite/cases.typ"
#include "compiler-suite/class.typ"
#include "compiler-suite/delimited.typ"
#include "compiler-suite/equation.typ"
#include "compiler-suite/frac.typ"
#include "compiler-suite/mat.typ"
#include "compiler-suite/multiline.typ"
#include "compiler-suite/op.typ"
#include "compiler-suite/primes.typ"
#include "compiler-suite/root.typ"
#include "compiler-suite/size.typ"
#include "compiler-suite/spacing.typ"
#include "compiler-suite/stretch.typ"
#include "compiler-suite/style.typ"
#include "compiler-suite/symbols.typ"
#include "compiler-suite/text.typ"
#include "compiler-suite/underover.typ"
#include "compiler-suite/vec.typ"
// TODO(28)
// #include "compiler-suite/cancel.typ"
// #include "compiler-suite/interactions.typ"
