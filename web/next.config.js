/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static export for Vercel Edge deployment
  output: 'standalone',

  // Webpack configuration for ONNX Runtime WebAssembly
  webpack: (config, { isServer }) => {
    // ONNX Runtime Web requires these settings
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };

    // Handle .wasm files
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    // Copy ONNX Runtime WASM files to public
    if (!isServer) {
      config.module.rules.push({
        test: /\.wasm$/,
        type: 'asset/resource',
      });
    }

    return config;
  },

  // Headers for WASM MIME type
  async headers() {
    return [
      {
        source: '/:path*.wasm',
        headers: [
          {
            key: 'Content-Type',
            value: 'application/wasm',
          },
        ],
      },
    ];
  },

  // Optimize images
  images: {
    unoptimized: true,
  },

  // TypeScript strict mode
  typescript: {
    ignoreBuildErrors: false,
  },

  // ESLint during build
  eslint: {
    ignoreDuringBuilds: false,
  },
};

module.exports = nextConfig;
