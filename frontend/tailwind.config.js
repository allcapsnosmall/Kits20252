/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#20243a",
        lavender: "#eeeaff",
        accent: "#6c63e8",
      },
    },
  },
  plugins: [],
};
