
#import "../packages/dotenv.typ": parse-env

// todo: what if I would like use other configuration like `.env-production`
#let env = parse-env(read("/.env"))

// Gets url-base from the `.env` file
#let url-base = {
  let url-base = env.at("URL_BASE", default: "")
  if not url-base.ends-with("/") {
    url-base = url-base + "/"
  }
  url-base
}
