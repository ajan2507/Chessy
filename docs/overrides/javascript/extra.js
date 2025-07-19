<<<<<<< HEAD
// Function that applies light/dark theme based on the user's preference
const applyAutoTheme = () => {
  // Determine the user's preferred color scheme
  const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  // Apply the appropriate attributes based on the user's preference
  if (prefersLight) {
    document.body.setAttribute("data-md-color-scheme", "default");
    document.body.setAttribute("data-md-color-primary", "indigo");
  } else if (prefersDark) {
    document.body.setAttribute("data-md-color-scheme", "slate");
    document.body.setAttribute("data-md-color-primary", "black");
  }
};

// Function that checks and applies light/dark theme based on the user's preference (if auto theme is enabled)
function checkAutoTheme() {
  // Array of supported language codes -> each language has its own palette (stored in local storage)
  const supportedLangCodes = ["en", "zh", "ko", "ja", "ru", "de", "fr", "es", "pt"];
  // Get the URL path
  const path = window.location.pathname;
  // Extract the language code from the URL (assuming it's in the format /xx/...)
  const langCode = path.split("/")[1];
  // Check if the extracted language code is in the supported languages
  const isValidLangCode = supportedLangCodes.includes(langCode);
  // Construct the local storage key based on the language code if valid, otherwise default to the root key
  const localStorageKey = isValidLangCode ? `/${langCode}/.__palette` : "/.__palette";
  // Retrieve the palette from local storage using the constructed key
  const palette = localStorage.getItem(localStorageKey);
  if (palette) {
    // Check if the palette's index is 0 (auto theme)
    const paletteObj = JSON.parse(palette);
    if (paletteObj && paletteObj.index === 0) {
      applyAutoTheme();
    }
  }
}

// Run function when the script loads
checkAutoTheme();

// Re-run the function when the user's preference changes (when the user changes their system theme)
window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", checkAutoTheme);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", checkAutoTheme);

// Re-run the function when the palette changes (e.g. user switched from dark theme to auto theme)
// ! We can't use window.addEventListener("storage", checkAutoTheme) because it will NOT be triggered on the current tab
// ! So we have to use the following workaround:
// Get the palette input for auto theme
var autoThemeInput = document.getElementById("__palette_1");
if (autoThemeInput) {
  // Add a click event listener to the input
  autoThemeInput.addEventListener("click", function () {
    // Check if the auto theme is selected
    if (autoThemeInput.checked) {
      // Re-run the function after a short delay (to ensure that the palette has been updated)
      setTimeout(applyAutoTheme);
    }
  });
}

// Add iframe navigation
window.onhashchange = function() {
    window.parent.postMessage({
        type: 'navigation',
        hash: window.location.pathname + window.location.search + window.location.hash
    }, '*');
};
=======
// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// Apply theme colors based on dark/light mode
const applyTheme = (isDark) => {
  document.body.setAttribute(
    "data-md-color-scheme",
    isDark ? "slate" : "default",
  );
  document.body.setAttribute(
    "data-md-color-primary",
    isDark ? "black" : "indigo",
  );
};

// Check and apply appropriate theme based on system/user preference
const checkTheme = () => {
  const palette = JSON.parse(localStorage.getItem(".__palette") || "{}");
  if (palette.index === 0) {
    // Auto mode is selected
    applyTheme(window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
};

// Watch for system theme changes
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", checkTheme);

// Initialize theme handling on page load
document.addEventListener("DOMContentLoaded", () => {
  // Watch for theme toggle changes
  document
    .getElementById("__palette_1")
    ?.addEventListener(
      "change",
      (e) => e.target.checked && setTimeout(checkTheme),
    );
  // Initial theme check
  checkTheme();
});

// Inkeep --------------------------------------------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  const enableSearchBar = true;

  const inkeepScript = document.createElement("script");
  inkeepScript.src =
    "https://cdn.jsdelivr.net/npm/@inkeep/cxkit-js@0.5/dist/embed.js";
  inkeepScript.type = "module";
  inkeepScript.defer = true;
  document.head.appendChild(inkeepScript);

  if (enableSearchBar) {
    const containerDiv = document.createElement("div");
    containerDiv.style.transform = "scale(0.7)";
    containerDiv.style.transformOrigin = "left center";

    const inkeepDiv = document.createElement("div");
    inkeepDiv.id = "inkeepSearchBar";
    containerDiv.appendChild(inkeepDiv);

    const headerElement = document.querySelector(".md-header__inner");
    const searchContainer = headerElement.querySelector(".md-header__source");

    if (headerElement && searchContainer) {
      headerElement.insertBefore(containerDiv, searchContainer);
    }
  }

  // Configuration object for Inkeep
  const config = {
    baseSettings: {
      apiKey: "13dfec2e75982bc9bae3199a08e13b86b5fbacd64e9b2f89",
      primaryBrandColor: "#E1FF25",
      organizationDisplayName: "Ultralytics",
      colorMode: {
        enableSystem: true,
      },
      theme: {
        styles: [
          {
            key: "main",
            type: "link",
            value: "/stylesheets/style.css",
          },
          {
            key: "chat-button",
            type: "style",
            value: `
              /* Light mode styling */
              .ikp-chat-button__button {
                background-color: #E1FF25;
                color: #111F68;
              }
              /* Dark mode styling */
              [data-theme="dark"] .ikp-chat-button__button {
                background-color: #40434f;
                color: #ffffff;
              }
              .ikp-chat-button__container {
                position: fixed;
                right: 1rem;
                bottom: 3rem;
              }
            `,
          },
        ],
      },
    },
    searchSettings: {
      placeholder: "Search",
    },
    aiChatSettings: {
      chatSubjectName: "Ultralytics",
      aiAssistantAvatar:
        "https://storage.googleapis.com/organization-image-assets/ultralytics-botAvatarSrcUrl-1729379860806.svg",
      exampleQuestions: [
        "What's new in Ultralytics YOLO11?",
        "How can I get started with Ultralytics HUB?",
        "How does Ultralytics Enterprise Licensing work?",
      ],
      getHelpOptions: [
        {
          name: "Ask on Ultralytics GitHub",
          icon: {
            builtIn: "FaGithub",
          },
          action: {
            type: "open_link",
            url: "https://github.com/ultralytics/ultralytics",
          },
        },
        {
          name: "Ask on Ultralytics Discourse",
          icon: {
            builtIn: "FaDiscourse",
          },
          action: {
            type: "open_link",
            url: "https://community.ultralytics.com/",
          },
        },
        {
          name: "Ask on Ultralytics Discord",
          icon: {
            builtIn: "FaDiscord",
          },
          action: {
            type: "open_link",
            url: "https://discord.com/invite/ultralytics",
          },
        },
      ],
    },
  };

  // Initialize Inkeep widgets when script loads
  inkeepScript.addEventListener("load", () => {
    const widgetContainer = document.getElementById("inkeepSearchBar");

    Inkeep.ChatButton(config);
    widgetContainer && Inkeep.SearchBar("#inkeepSearchBar", config);
  });
});

// Fix language switcher links
(function () {
  function fixLanguageLinks() {
    const path = location.pathname;
    const links = document.querySelectorAll(".md-select__link");
    if (!links.length) return;

    const langs = [];
    let defaultLink = null;

    // Extract language codes
    links.forEach((link) => {
      const href = link.getAttribute("href");
      if (!href) return;

      const url = new URL(href, location.origin);
      const match = url.pathname.match(/^\/([a-z]{2})\/?$/);

      if (match) langs.push({ code: match[1], link });
      else if (url.pathname === "/" || url.pathname === "") defaultLink = link;
    });

    // Find current language and base path
    let basePath = path;
    for (const lang of langs) {
      if (path.startsWith("/" + lang.code + "/")) {
        basePath = path.substring(lang.code.length + 1);
        break;
      }
    }

    // Update links
    langs.forEach(
      (lang) => (lang.link.href = location.origin + "/" + lang.code + basePath),
    );
    if (defaultLink) defaultLink.href = location.origin + basePath;
  }

  // Run immediately
  fixLanguageLinks();

  // Handle SPA navigation
  if (typeof document$ !== "undefined") {
    document$.subscribe(() => setTimeout(fixLanguageLinks, 50));
  } else {
    let lastPath = location.pathname;
    setInterval(() => {
      if (location.pathname !== lastPath) {
        lastPath = location.pathname;
        setTimeout(fixLanguageLinks, 50);
      }
    }, 200);
  }
})();
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
