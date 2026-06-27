#let _fallback-elem(tag, ..args) = {
  let pos = args.pos()
  let named = args.named()

  if "body" in named {
    named.at("body")
  } else if pos.len() > 0 {
    pos.last()
  } else {
    none
  }
}

#let _std-html = dictionary(std).at("html", default: none)

#let elem = if _std-html == none {
  _fallback-elem
} else {
  _std-html.elem
}

#let frame = if _std-html == none {
  body => body
} else {
  _std-html.frame
}
