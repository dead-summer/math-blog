#import "@preview/tidy:0.4.1"
#import "@preview/codly:1.3.0"
#import "../src/lib.typ" as mathyml

#show: codly.codly-init
#codly.codly(number-format: none, languages: codly.typst-icon)

#set heading(numbering: (..num) => if num.pos().len() < 4 {
  numbering("1.1.", ..num)
})
#show heading.where(level: 1, outlined: true): inner => [
  #{
    set text(size: 24pt)
    inner
  }
]

#set document(title: "mathyml documentation")
#show link: it => underline(text(fill: blue, it))

#let package-toml = toml("../typst.toml").package
#let version = [v#package-toml.version]

#align(center + horizon)[
  #set text(size: 20pt)
  #block(text(weight: 700, size: 40pt, "mathyml"))
  #package-toml.description
  #v(40pt)
  #version
  #h(40pt)
  #datetime.today().display()

  #link(package-toml.repository)
  #v(5pt)
  #package-toml.authors.join("  ")
]

#pagebreak(weak: true)
#set page(
  numbering: "1/1",
  header: grid(
    columns: (auto, 1fr, auto),
    align(left)[mathyml], [], align(right, version),
  ),
)

#set text(size: 12pt)

#outline(indent: 0.3em, depth: 3)

#pagebreak(weak: true)

#let module(name, path, private: false, do_pagebreak: true) = {
  let docs = tidy.parse-module(
    read(path),
    scope: (mathyml: mathyml,),
    require-all-parameters: not private,
  )
  if do_pagebreak {
    pagebreak(weak: true)
  }
  [== #name]
  tidy.show-module(
    docs,
    omit-private-definitions: true,
    first-heading-level: 2,
    show-outline: false,
  )
}

= Overview

Mathyml converts your typst equations to MathML Core so that they work well with HTML export.
MathML Core is a language for describing mathematical notation and supported by major browsers (firefox and chrome).
You can find an overview of MathML on #link("https://developer.mozilla.org/en-US/docs/Web/MathML")[mdn] and in the #link("https://www.w3.org/TR/mathml-core/")[specification].

Note that MathML rendering is certainly not perfect, some features work better and some worse.
In general the output tends to look much better in firefox than chrome.
See the #link(<errors>)[section about missing/ non-working features].

MathML Core is not complete and can't render everything itself. Instead it relies on Web Platform features (such as CSS) (see the #link("https://w3c.github.io/mathml-core/docs/explainer.html")[explainer]).
#link(<list-of-non-core>)[Here] is a list of polyfills/ features used that do not come from MathML Core.

= Quickstart<quickstart>

First, import mathyml and include the prelude, which defines replacements for elements which mathyml can't handle (e.g. `bold` or `cal`).
```typ
#import "@local/mathyml:0.1.0"
#import mathyml: to-mathml
#import mathyml.prelude: *
```
Include the required stylesheet (and the mathfont):
```typ
#mathyml.stylesheets()
```
Note that the mathfont is required, else the rendering looks really bad. The font is currently downloaded from #link("https://github.com/fred-wang/MathFonts")[github]. I would recommend changing the font family to your liking and downloading the css files yourself (so that it works without an internet connection).

Convert equations manually (but only for html):
```typ
The fraction #to-mathml($1/3$) is not a decimal number. And we know
#to-mathml($ a^2 + b^2 = c^2. $)
```
You can also convert equations automatically.
If this panics, try `try-to-mathml` instead, which will create a svg on error.
```typ
#show math.equation: to-mathml

To solve the cubic equation $t^3 + p t + q = 0$ (where the real numbers
$p, q$ satisfy $4p^3 + 27q^2 > 0$) one can use Cardano's formula:
$
  root(3, -q/2 + sqrt(q^2/4 + p^3/27)) + root(3, -q/2 - sqrt(q^2/4 + p^3/27)).
$
```

#pagebreak(weak: true)

= Missing/ non-working features<errors>
The list below comes from own testing and typst's test suite.
I've indicated the corresponding test names in parentheses.
#link("https://akida.codeberg.page/mathyml")[Here] you can see the html output for the examples and tests.

#let e-heading = heading.with(level: 2, outlined: false, numbering: none)

#e-heading[Not working/ looking not perfect]
/ set and show rules: Examples:
  - `#show math.equation: set align(start)`
  - `#show math.equation: set text(font: "STIX Two Math")`
  (E.g. issue-3973-math-equation-align, math-attach-show-limit, math-cases-delim, math-equation-show-rule, math-mat-delim-set,math-mat-augment-set, math-vec-delim-set)
/ non-math elements: generally not supported
/ cancel: (just not implemented)
/ multiline: break in text (issue-1948-math-text-break)
/ weak spacing: (math-lr-weak-spacing)
/ equation numbering and labels: (math-equation-align-numbered)
/ rtl: (issue-3696-equation-rtl)
/ semantics:
  mathml allows adding svg annotations (https://www.w3.org/TR/mathml-core/#semantics-and-presentation).
  Typst html elements may not contain hyphens, so it is currently not possible to create an `annotation-xml` element.
/ accents:
  - chrome:
    - it generally seems the chrome does not recognize the width of the accents, so they also collide and are offset (math-accent-high-base, math-accent-sym-call)
    - dotless (math-accent-dotless)
  - `size` does not work
/ alignment:
  - nested alignments are unsupported
  - chrome: sometimes chrome does not align the columns correctly (math-align-toggle, math-align-wider-first-column) (does not respect `text-align: right;`)
/ attach:
  - `t` and `b` attachments are further away (math-attach-mixed, math-attach-subscript-multiline)
  - scripts are limits sometimes  (math-op-scripts-vs-limits) (I am not sure to improve this, as displaystyle should be enough)
  - limits are scripts sometimes (math-accent-wide-base, math-attach-limit-long) (set `movablelimits="false"` for the next inner `mo` in `limits`)
  - attached text is not rendered as stack above but at the top right (math-stretch-attach-nested-equation)
  - nested attachs are not merged (math-attach-nested-deep-base)
  - attachs are not stretched automatically
/ lr/ mid:
  - the size parameter does not work 
  - firefox: mid is sometimes not completely large enough (math-lr-mid)
  - firefox: lr does not include subscript (math-lr-unparen) 
/ mat:
  - chrome: gap does not work (math-mat-gaps)
  - augment colors (math-mat-augment)
  - linebreaks are not discarded (math-mat-linebreaks)
  - manual alignment does not work (math-mat-align)
/ vec:
  - manual alignment does not work (math-vec-align-explicit-alternating)
/ primes:
  chrome: they are really small (math-primes-attach)
/ sqrt:
  chrome: small artifacts (math-root-large-body, math-root-large-index)
/ sizes (display, inline, script, sscript):
  - only two levels (math-size)
  - only available via prelude
  - `cramped` is not supported
/ variants (serif, sans, frak, mono, bb, cal):
  - only available via prelude
/ styles (upright, italic, bold):
  - only available via prelude
  - dotless styles don't really work (math-style-dotless)
/ spacing:
  MathML adds extra spaces (math-spacing-kept-spaces) (not too bad)
/ stretch:
  - only supports `op`s
  - only works (horizontally) with an element above/ below (see math-stretch-complex, math-stretch-horizontal, math-underover-brace)
/ overset: multiline in overbrace looks weird, it has extra space


#e-heading[List of polyfills/ features used that do not come from MathML Core]<list-of-non-core>
- some CSS for inserting html `frame`s inline or as a block
- some CSS for alignment points
- cases: uses CSS `padding-bottom` for gap
- vec: uses CSS `padding-bottom` for gap, `text-align` for align.
- mat:
  - uses CSS `padding-bottom` and `padding-right` for gap, `text-align` for align.
  - uses CSS (`border-right` and `border-bottom`) for augments
- primes: uses CSS (`padding-left`)

#pagebreak(weak: true)

// = Examples
#let example(name, header, text: none, do-break: true) = {
  let file_content = read(name).split("\n").slice(3).join("\n")

  heading(level: 2, header)
  if text != none { text }

  set heading(outlined: false)

  counter(heading).update(0)
  tidy.show-example.show-example(
    raw(file_content),
    scale-preview: 100%,
    mode: "markup",
    scope: (mathyml: mathyml),
    ratio: 1.2
  )

  if do-break {
    pagebreak(weak: true)
  }
}

// #example("../examples/quickstart.typ", "Quickstart", do-break: false)

#pagebreak(weak: true)
#counter(heading).update(1)

= Reference

#module("lib", "../src/lib.typ", do_pagebreak: false)
#module("prelude", "../src/prelude.typ", do_pagebreak: false)
