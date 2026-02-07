
#import "../packages/zebraw.typ": *
#import "@preview/shiroa:0.2.3": is-web-target, is-pdf-target, plain-text, is-html-target, templates
#import templates: *
#import "mod.typ": *
#import "theme.typ": *
#import "math-macros.typ": *
#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/theorion:0.4.0": *
#import cosmos.fancy: *
#import "@preview/numbly:0.1.0": numbly

// Metadata
#let is-html-target = is-html-target()
#let is-pdf-target = is-pdf-target()
#let is-web-target = is-web-target() or sys-is-html-target
#let is-md-target = target == "md"
#let sys-is-html-target = ("target" in dictionary(std))

#let default-kind = "post"

#let build-kind = sys.inputs.at("build-kind", default: default-kind)

#let text-fonts = (
  "Libertinus Serif",
  // todo: exclude it if language is not Chinese.
  "Source Han Serif SC",
)

#let code-font = (
  "DejaVu Sans Mono",
)

// Sizes
#let main-size = if sys-is-html-target {
  16pt
} else {
  10.5pt
}
// ,
#let heading-sizes = (22pt, 18pt, 14pt, 12pt, main-size)
#let list-indent = 0.5em

/// Creates an embedded block typst frame.
#let div-frame(content, attrs: (:), tag: "div") = html.elem(tag, html.frame(content), attrs: attrs)
#let span-frame = div-frame.with(tag: "span")
#let p-frame = div-frame.with(tag: "p")

// defaults
#let (
  style: theme-style,
  is-dark: is-dark-theme,
  is-light: is-light-theme,
  main-color: main-color,
  dash-color: dash-color,
  code-extra-colors: code-extra-colors,
) = default-theme

#let markup-rules(body, lang: none, region: none) = {
  set text(lang: lang) if lang != none
  set text(region: region) if region != none
  set text(font: text-fonts)

  set text(main-size) if sys-is-html-target
  set text(fill: rgb("dfdfd6")) if is-dark-theme and sys-is-html-target
  show link: set text(fill: dash-color)
  // 将引用转换为 HTML 超链接
  show ref: it => context if sys-is-html-target {
    let target-label = it.target
    let label-id = "label-" + str(make-unique-label(
      str(target-label),
      disambiguator: label-disambiguator.at(it.location()).at(str(target-label), default: 0) + 1,
    ))
    html.elem(
      "a",
      attrs: ("href": "#" + label-id, "class": "ref-link"),
      { set text(fill: dash-color); it }
    )
  } else {
    set text(fill: dash-color)
    it
  }

  show heading: it => {
    set text(size: heading-sizes.at(it.level))

    block(
      spacing: 0.7em * 1.5 * 1.2,
      below: 0.7em * 1.2,
      {
        if is-web-target {
          show link: static-heading-link(it)
          heading-hash(it, hash-color: dash-color)
        }

        it
      },
    )
  }
  set heading(numbering: numbly("{1:1}", default: "1.1"))

  body
}

#let equation-rules(body) = {
  show math.equation: set text(weight: 400)
  show math.equation.where(block: true): it => context if shiroa-sys-target() == "html" {
    // 为带标签的方程式生成 HTML id，使引用可以链接到此处
    let label-id = if it.has("label") {
      let lbl = it.label
      "label-" + str(make-unique-label(
        str(lbl),
        disambiguator: label-disambiguator.at(it.location()).at(str(lbl), default: 0) + 1,
      ))
    } else { none }

    // theme-frame 会渲染 dark/light 两个版本，id 必须放在外层避免重复
    let content = theme-frame(
      tag: "div",
      theme => {
        set text(fill: theme.main-color)
        p-frame(
          attrs: (
            "class": "block-equation",
            "role": "math",
          ),
          it,
        )
      },
    )

    if label-id != none {
      html.elem("div", content, attrs: ("id": label-id, "data-typst-label": label-id))
    } else {
      content
    }
  } else {
    it
  }
  show math.equation.where(block: false): it => context if shiroa-sys-target() == "html" {
    theme-frame(
      tag: "span",
      theme => {
        set text(fill: theme.main-color)
        span-frame(attrs: (class: "inline-equation"), it)
      },
    )
  } else {
    it
  }

  set math.equation(numbering: "(1)")
  // 有 label 时才编号
  show math.equation.where(block: true): it => {
    if not it.has("label") {
      let fields = it.fields()
      let _ = fields.remove("body")
      fields.numbering = none
      [#counter(math.equation).update(v => v - 1)#math.equation(..fields, it.body)<math-equation-without-label>]
    } else {
      it
    }
  }

  body
}

#let code-block-rules(body) = {
  let init-with-theme((code-extra-colors, is-dark)) = if is-dark {
    zebraw-init.with(
      // should vary by theme
      background-color: if code-extra-colors.bg != none {
        (code-extra-colors.bg, code-extra-colors.bg)
      },
      highlight-color: rgb("#3d59a1"),
      comment-color: rgb("#394b70"),
      lang-color: rgb("#3d59a1"),
      lang: false,
      numbering: false,
    )
  } else {
    zebraw-init.with(
      // should vary by theme
      background-color: if code-extra-colors.bg != none {
        (code-extra-colors.bg, code-extra-colors.bg)
      },
      lang: false,
      numbering: false,
    )
  }

  /// HTML code block supported by zebraw.
  show: init-with-theme(default-theme)


  let mk-raw(
    it,
    tag: "div",
    inline: false,
  ) = theme-frame(
    tag: tag,
    theme => {
      show: init-with-theme(theme)
      let code-extra-colors = theme.code-extra-colors
      let use-fg = not inline and code-extra-colors.fg != none
      set text(fill: code-extra-colors.fg) if use-fg
      set text(fill: if theme.is-dark { rgb("dfdfd6") } else { black }) if not use-fg
      set raw(theme: theme-style.code-theme) if theme.style.code-theme.len() > 0
      set par(justify: false)
      zebraw(
        block-width: 100%,
        // line-width: 100%,
        wrap: false,
        it,
      )
    },
  )

  show raw: set text(font: code-font)
  show raw.where(block: false): it => context if shiroa-sys-target() == "paged" {
    it
  } else {
    mk-raw(it, tag: "span", inline: true)
  }
  show raw.where(block: true): it => context if shiroa-sys-target() == "paged" {
    set raw(theme: theme-style.code-theme) if theme-style.code-theme.len() > 0
    rect(
      width: 100%,
      inset: (x: 4pt, y: 5pt),
      radius: 4pt,
      fill: code-extra-colors.bg,
      [
        #set text(fill: code-extra-colors.fg) if code-extra-colors.fg != none
        #set par(justify: false)
        // #place(right, text(luma(110), it.lang))
        #it
      ],
    )
  } else {
    mk-raw(it)
  }
  body
}

#let visual-rules(body) = {
  import "env.typ": url-base
  // Resolves the path to the image source
  let resolve(path) = (
    path.replace(
      // Substitutes the paths with some assumption.
      // In the astro sites, the assets are store in `public/` directory.
      regex("^[./]*/public/"),
      url-base,
    )
  )

  show image: it => context if shiroa-sys-target() == "paged" {
    it
  } else {
    html.elem("img", attrs: (src: resolve(it.source)))
  }

  body
}

#let shared-template(
  title: "Untitled",
  author: "Unknown",
  desc: [This is a blog post.],
  date: "2024-08-15",
  tags: (),
  kind: "post",
  lang: none,
  region: none,
  show-outline: true,
  body,
) = {
  let is-same-kind = build-kind == kind

  show: it => if is-same-kind {
    // set basic document metadata
    set document(
      author: (author,),
      ..if not is-web-target { (title: title) },
    )

    // markup setting
    show: markup-rules.with(
      lang: lang,
      region: region,
    )
    // math setting
    show: equation-rules
    // code block setting
    show: code-block-rules
    // visualization setting
    show: visual-rules

    show: it => if sys-is-html-target {
      show footnote: it => context {
        let num = counter(footnote).get().at(0)
        link(label("footnote-" + str(num)), super(str(num)))
      }

      it
    } else {
      it
    }

    // Main body.
    set par(justify: true)
    it
  } else {
    it
  }

  if is-same-kind [
    #metadata((
      title: plain-text(title),
      author: plain-text(author),
      description: plain-text(desc),
      date: date,
      tags: tags,
      lang: lang,
      region: region,
    )) <frontmatter>
  ]

  context if show-outline and is-same-kind and sys-is-html-target {
    if query(heading).len() == 0 {
      return
    }

    let outline-counter = counter("html-outline")
    outline-counter.update(0)
    show outline.entry: it => html.elem(
      "div",
      attrs: (
        class: "outline-item x-heading-" + str(it.level),
      ),
      {
        outline-counter.step(level: it.level)
        static-heading-link(it.element, body: [#sym.section#context outline-counter.display("1.") #it.element.body])
      },
    )
    html.elem(
      "div",
      attrs: (
        class: "outline",
      ),
      outline(title: none),
    )
    html.elem("hr")
  }

  body

  context if is-same-kind and sys-is-html-target {
    query(footnote)
      .enumerate()
      .map(((idx, it)) => {
        enum.item[
          #html.elem(
            "div",
            attrs: ("data-typst-label": "footnote-" + str(idx + 1)),
            it.body,
          )
        ]
      })
      .join()
  }
}
