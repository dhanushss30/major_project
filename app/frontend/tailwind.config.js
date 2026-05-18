/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Research-lab dark palette
        ink: {
          900: '#0a0e14',
          800: '#0f1419',
          700: '#161b22',
          600: '#1e2430',
          500: '#2a3340',
          400: '#3b4452',
        },
        accent: {
          cyan:   '#00f0ff',
          lime:   '#a3ff00',
          amber:  '#ffb800',
          rose:   '#ff6b6b',
          violet: '#a78bfa',
        },
        muted: {
          DEFAULT: '#7a8290',
          light:   '#9da6b5',
          dark:    '#5b6472',
        },
      },
      fontFamily: {
        mono:    ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
        display: ['Space Grotesk', 'Inter', 'system-ui', 'sans-serif'],
        body:    ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow':    'pulse 3s ease-in-out infinite',
        'glow':          'glow 2s ease-in-out infinite alternate',
        'fade-in':       'fadeIn 0.5s ease-in-out',
        'slide-up':      'slideUp 0.4s ease-out',
        'shimmer':       'shimmer 2s linear infinite',
      },
      keyframes: {
        glow: {
          '0%':   { boxShadow: '0 0 5px rgba(0,240,255,0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(0,240,255,0.9)' },
        },
        fadeIn: {
          '0%':   { opacity: 0 },
          '100%': { opacity: 1 },
        },
        slideUp: {
          '0%':   { opacity: 0, transform: 'translateY(10px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
