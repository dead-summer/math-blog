# mathyml
[mathyml](https://codeberg.org/akida/mathyml) converts your equations to MathML so that they work well with HTML export.
See the [manual](./docs/doc.pdf) for documentation.

## Overview
Mathyml converts your typst equations to MathML Core.
MathML Core is a language for describing mathematical notation and supported by major browsers (firefox and chrome).
You can find an overview of MathML on [mdn](https://developer.mozilla.org/en-US/docs/Web/MathML) and in the [specification](https://www.w3.org/TR/mathml-core/).

Note that MathML rendering is certainly not perfect, some features work better and some worse.
In general the output tends to look much better in firefox than chrome.
See the section about missing/ non-working features below.

MathML Core is not complete and can't render everything itself. Instead it relies on Web Platform features (such as CSS) (see the [explainer](https://w3c.github.io/mathml-core/docs/explainer.html)).
You can find a list of of elements/ features used that do not come from MathML Core below.

## Installation
Execute `python install.py`. This will install the `mathyml` package to the local package folder.

## Quickstart
First, import mathyml and include the prelude, which defines replacements for elements which mathyml can't handle (e.g. `bold` or `cal`).
```typst
#import "@local/mathyml:0.1.0"
#import mathyml: to-mathml
#import mathyml.prelude: *
```
Include the required stylesheet (and the mathfont):
```typst
#mathyml.stylesheets()
```
Note that the mathfont is required, else the rendering looks really bad. The font is currently downloaded from [github](https://github.com/fred-wang/MathFonts). I would recommend changing the font family to your liking and downloading the css files yourself (so that it works without an internet connection).

Convert equations manually:
```typst
The fraction #to-mathml($1/3$) is not a decimal number. And we know
#to-mathml($ a^2 + b^2 = c^2. $)
```
You can also convert equations automatically.
If this panics, try `try-to-mathml` instead, which will create a svg on error.
```typst
#show math.equation: to-mathml

To solve the cubic equation $t^3 + p t + q = 0$ (where the real numbers
$p, q$ satisfy $4p^3 + 27q^2 > 0$) one can use Cardano's formula:
$
  root(3, -q/2 + sqrt(q^2/4 + p^3/27)) + root(3, -q/2 - sqrt(q^2/4 + p^3/27)).
$
```

## Missing/ non-working features
You can find a list of all missing/ non-working features in the [manual](./docs/doc.pdf).
[Here](https://akida.codeberg.page/mathyml) you can see the html output for the examples and tests.
