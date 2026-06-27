
#import "/typ/templates/html.typ" as html
#import "/typ/templates/mod.typ": sys-is-html-target

#let en(body) = {
  show: text.with(lang: "en")
  context if std.target() == "html" {
    html.elem("p", attrs: ("lang": "en"), body)
  } else {
    body
  }
}

#let zh(body) = {
  show: text.with(lang: "zh")
  context if std.target() == "html" {
    html.elem("p", attrs: ("lang": "zh", "translate": "no"), body)
  } else {
    body
  }
}

#en[
  Welcome to my blog.
]
