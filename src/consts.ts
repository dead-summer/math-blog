// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

import * as config from "astro:env/client";

/**
 * Whether to enable theming (dark & light mode).
 */
export const kEnableTheming = true;
/**
 * Whether to enable post search (needs Js).
 */
export const kEnableSearch = true;

/**
 * The title of the website.
 */
export const kSiteTitle: string = config.SITE_TITLE || "My Blog";

/**
 * The title of the website.
 */
export const kSiteLogo: string = kSiteTitle;
/**
 * The title of the website, used in the index page.
 */
export const kSiteIndexTitle: string = config.SITE_INDEX_TITLE || kSiteTitle;
/**
 * The description of the website.
 */
export const kSiteDescription: string = config.SITE_DESCRIPTION || "My blog.";
/**
 * The name of the site owner.
 */
export const kSiteOwner: string = config.SITE_OWNER || "Blog Owner";
/**
 * The source code URL of the site.
 *
 * Disable this if you don't want to show the source code link.
 */
export const kSiteSourceUrl: string | undefined = config.SITE_SOURCE_URL;
/**
 * The baidu verification code, used for SEO.
 */
export const kBaiduVeriCode: string | undefined =
  config.BAIDU_VERIFICATION_CODE;

/**
 * The URL base of the website.
 * - For a GitHub page `https://username.github.io/repo`, the URL base is `/repo/`.
 * - For a netlify page, the URL base is `/`.
 */
export const kUrlBase = (config.URL_BASE || "").replace(/\/$/, "");
