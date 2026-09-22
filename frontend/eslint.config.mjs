import nextConfig from "eslint-config-next";

const config = [
  ...nextConfig,
  {
    ignores: ["node_modules/", ".next/", "out/", "*.config.*"],
    rules: {
      "eslint-comments/no-unused-disable": "off",
    },
  },
];

export default config;