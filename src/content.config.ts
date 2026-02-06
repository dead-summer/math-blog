import { glob } from "astro/loaders";
import { defineCollection, z } from "astro:content";

const nullToUndefined = <T>(value: T | null) => value ?? undefined;
const coerceStringOpt = () => z.string().nullish().transform(nullToUndefined);

const blog = defineCollection({
  loader: glob({ base: "./content/article", pattern: "*.typ" }),
  schema: z.object({
    title: z.string(),
    lang: coerceStringOpt(),
    region: coerceStringOpt(),
    author: z.string().optional(),
    description: z.any().optional(),
    date: z.coerce.date(),
    tags: z.array(z.string()).optional(),
  }),
});

export const collections = { blog };
