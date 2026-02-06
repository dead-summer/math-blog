import type { APIContext } from "astro";
import rss, { type RSSFeedItem } from "@astrojs/rss";
import { getCollection } from "astro:content";

import { kUrlBase, kSiteTitle, kSiteDescription } from "$consts";
import { published } from "$content";

export async function GET(context: APIContext) {
  if (!context.site) {
    throw new Error("No site URL found");
  }

  const items = (await getCollection("blog")).filter(published).map(
    (item): RSSFeedItem => ({
      title: item.data.title,
      description: item.data.description,
      pubDate: item.data.date,
      categories: item.data.tags,
      link: `${kUrlBase}/article/${item.id}/`,
    })
  );

  return rss({
    title: kSiteTitle,
    description: kSiteDescription,
    site: context.site,
    items,
  });
}
