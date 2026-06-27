#import "@preview/shiroa:0.2.3": templates, book-sys
#import templates: *
#import "html.typ" as html

#let is-md-target = book-sys.target == "md"
#let sys-is-html-target = book-sys.is-html-target

// Theme (Colors)
#let dark-theme = book-theme-from(toml("theme-style.toml"), xml: it => xml(it), target: "web-ayu")
#let light-theme = book-theme-from(
  toml("theme-style.toml"),
  xml: it => xml(it),
  target: "web-light",
)
#let paged-theme = book-theme-from(
  toml("theme-style.toml"),
  xml: it => xml(it),
  target: "pdf",
)
#let default-theme = if sys-is-html-target {
  dark-theme
} else {
  paged-theme
}

#let theme-frame(render, tag: "div", class: none, theme-tag: none) = context if is-md-target {
  show: html.elem.with(tag)
  show: html.elem.with("picture")
  html.elem(
    "m1source",
    attrs: (media: "(prefers-color-scheme: dark)"),
    render(dark-theme),
  )
  render(light-theme)
} else if std.target() == "html" {
  let actual-theme-tag = if theme-tag == none { tag } else { theme-tag }
  html.elem(
    tag,
    attrs: (class: "code-image themed" + if class != none { " " + class }),
    {
      html.elem(
        actual-theme-tag,
        render(dark-theme),
        attrs: (class: "dark"),
      )
      html.elem(
        actual-theme-tag,
        render(light-theme),
        attrs: (class: "light"),
      )
    },
  )
} else {
  render(default-theme)
}
