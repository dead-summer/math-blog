#import "shared.typ": *
// The default template for blog posts.
#let main = shared-template.with(kind: "post", lang: "en")
// shortcut for English blog posts
#let main-en = main
// shortcut for Chinese blog posts
#let main-zh = main.with(lang: "zh", region: "cn")
