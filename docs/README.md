<<<<<<< HEAD
# Ultralytics Docs

Ultralytics Docs are deployed to [https://docs.ultralytics.com](https://docs.ultralytics.com).

[![pages-build-deployment](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment)  [![Check Broken links](https://github.com/ultralytics/docs/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/links.yml)

## Install Ultralytics package

[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

To install the ultralytics package in developer mode, you will need to have Git and Python 3 installed on your system. Then, follow these steps:

1. Clone the ultralytics repository to your local machine using Git:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2. Navigate to the root directory of the repository:

    ```bash
    cd ultralytics
    ```

3. Install the package in developer mode using pip:

    ```bash
    pip install -e '.[dev]'
    ```

This will install the ultralytics package and its dependencies in developer mode, allowing you to make changes to the package code and have them reflected immediately in your Python environment.

Note that you may need to use the pip3 command instead of pip if you have multiple versions of Python installed on your system.

## Building and Serving Locally

The `mkdocs serve` command is used to build and serve a local version of the MkDocs documentation site. It is typically used during the development and testing phase of a documentation project.

```bash
mkdocs serve
```

Here is a breakdown of what this command does:

- `mkdocs`: This is the command-line interface (CLI) for the MkDocs static site generator. It is used to build and serve MkDocs sites.
- `serve`: This is a subcommand of the `mkdocs` CLI that tells it to build and serve the documentation site locally.
- `-a`: This flag specifies the hostname and port number to bind the server to. The default value is `localhost:8000`.
- `-t`: This flag specifies the theme to use for the documentation site. The default value is `mkdocs`.
- `-s`: This flag tells the `serve` command to serve the site in silent mode, which means it will not display any log messages or progress updates. When you run the `mkdocs serve` command, it will build the documentation site using the files in the `docs/` directory and serve it at the specified hostname and port number. You can then view the site by going to the URL in your web browser.

While the site is being served, you can make changes to the documentation files and see them reflected in the live site immediately. This is useful for testing and debugging your documentation before deploying it to a live server.

To stop the serve command and terminate the local server, you can use the `CTRL+C` keyboard shortcut.

## Building and Serving Multi-Language

For multi-language MkDocs sites use the following additional steps:

1. Add all new language *.md files to git commit: `git add docs/**/*.md -f`
2. Build all languages to the `/site` directory. Verify that the top-level `/site` directory contains `CNAME`, `robots.txt` and `sitemap.xml` files, if applicable.

    ```bash
    # Remove existing /site directory
    rm -rf site

    # Loop through all *.yml files in the docs directory
    mkdocs build -f docs/mkdocs.yml
    for file in docs/mkdocs_*.yml; do
      echo "Building MkDocs site with configuration file: $file"
      mkdocs build -f "$file"
    done
    ```

3. Preview in web browser with:

    ```bash
    cd site
    python -m http.server
    open http://localhost:8000  # on macOS
    ```

Note the above steps are combined into the Ultralytics [build_docs.py](https://github.com/ultralytics/ultralytics/blob/main/docs/build_docs.py) script.

## Deploying Your Documentation Site

To deploy your MkDocs documentation site, you will need to choose a hosting provider and a deployment method. Some popular options include GitHub Pages, GitLab Pages, and Amazon S3.

Before you can deploy your site, you will need to configure your `mkdocs.yml` file to specify the remote host and any other necessary deployment settings.

Once you have configured your `mkdocs.yml` file, you can use the `mkdocs deploy` command to build and deploy your site. This command will build the documentation site using the files in the `docs/` directory and the specified configuration file and theme, and then deploy the site to the specified remote host.

For example, to deploy your site to GitHub Pages using the gh-deploy plugin, you can use the following command:

```bash
mkdocs gh-deploy
```

If you are using GitHub Pages, you can set a custom domain for your documentation site by going to the "Settings" page for your repository and updating the "Custom domain" field in the "GitHub Pages" section.

![196814117-fc16e711-d2be-4722-9536-b7c6d78fd167](https://user-images.githubusercontent.com/26833433/210150206-9e86dcd7-10af-43e4-9eb2-9518b3799eac.png)

For more information on deploying your MkDocs documentation site, see the [MkDocs documentation](https://www.mkdocs.org/user-guide/deploying-your-docs/).
=======
<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 📚 Ultralytics Docs

Welcome to Ultralytics Docs, your comprehensive resource for understanding and utilizing our state-of-the-art [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) tools and models, including [Ultralytics YOLO](https://docs.ultralytics.com/models/yolov8/). These documents are actively maintained and deployed to [https://docs.ultralytics.com](https://docs.ultralytics.com/) for easy access.

[![pages-build-deployment](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/pages/pages-build-deployment)
[![Check Broken links](https://github.com/ultralytics/docs/actions/workflows/links.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/links.yml)
[![Check Domains](https://github.com/ultralytics/docs/actions/workflows/check_domains.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/check_domains.yml)
[![Ultralytics Actions](https://github.com/ultralytics/docs/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/docs/actions/workflows/format.yml)

<a href="https://discord.com/invite/ultralytics"><img alt="Discord" src="https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue"></a> <a href="https://community.ultralytics.com/"><img alt="Ultralytics Forums" src="https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue"></a> <a href="https://www.reddit.com/r/ultralytics/"><img alt="Ultralytics Reddit" src="https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue"></a>

## 🛠️ Installation

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
[![Downloads](https://static.pepy.tech/badge/ultralytics)](https://www.pepy.tech/projects/ultralytics)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

To install the `ultralytics` package in developer mode, which allows you to modify the source code directly, ensure you have [Git](https://git-scm.com/) and [Python](https://www.python.org/) 3.9 or later installed on your system. Then, follow these steps:

1.  Clone the `ultralytics` repository to your local machine using Git:

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    ```

2.  Navigate to the cloned repository's root directory:

    ```bash
    cd ultralytics
    ```

3.  Install the package in editable mode (`-e`) along with its development dependencies (`[dev]`) using [pip](https://pip.pypa.io/en/stable/):

    ```bash
    pip install -e '.[dev]'
    ```

    This command installs the `ultralytics` package such that changes to the source code are immediately reflected in your environment, ideal for development.

## 🚀 Building and Serving Locally

The `mkdocs serve` command builds and serves a local version of your [MkDocs](https://www.mkdocs.org/) documentation. This is highly useful during development and testing to preview changes.

```bash
mkdocs serve
```

- **Command Breakdown:**
    - `mkdocs`: The main MkDocs command-line interface tool.
    - `serve`: The subcommand used to build and locally serve your documentation site.
- **Note:**
    - `mkdocs serve` includes live reloading, automatically updating the preview in your browser as you save changes to the documentation files.
    - To stop the local server, simply press `CTRL+C` in your terminal.

## 🌍 Building and Serving Multi-Language

If your documentation supports multiple languages, follow these steps to build and preview all versions:

1.  Stage all new or modified language Markdown (`.md`) files using Git:

    ```bash
    git add docs/**/*.md -f
    ```

2.  Build all language versions into the `/site` directory. This script ensures that relevant root-level files are included and clears the previous build:

    ```bash
    # Clear existing /site directory to prevent conflicts
    rm -rf site

    # Build the default language site using the primary config file
    mkdocs build -f docs/mkdocs.yml

    # Loop through each language-specific config file and build its site
    for file in docs/mkdocs_*.yml; do
      echo "Building MkDocs site with $file"
      mkdocs build -f "$file"
    done
    ```

3.  To preview the complete multi-language site locally, navigate into the build output directory and start a simple [Python HTTP server](https://docs.python.org/3/library/http.server.html):
    ```bash
    cd site
    python -m http.server
    # Open http://localhost:8000 in your preferred web browser
    ```
    Access the live preview site at `http://localhost:8000`.

## 📤 Deploying Your Documentation Site

To deploy your MkDocs documentation site, choose a hosting provider and configure your deployment method. Common options include [GitHub Pages](https://pages.github.com/), GitLab Pages, or other static site hosting services.

- Configure deployment settings within your `mkdocs.yml` file.
- Use the `mkdocs deploy` command specific to your chosen provider to build and deploy your site.

* **GitHub Pages Deployment Example:**
  If deploying to GitHub Pages, you can use the built-in command:

    ```bash
    mkdocs gh-deploy
    ```

    After deployment, you might need to update the "Custom domain" settings in your repository's settings page if you wish to use a personalized URL.

    ![GitHub Pages Custom Domain Setting](https://user-images.githubusercontent.com/26833433/210150206-9e86dcd7-10af-43e4-9eb2-9518b3799eac.png)

- For detailed instructions on various deployment methods, consult the official [MkDocs Deploying your docs guide](https://www.mkdocs.org/user-guide/deploying-your-docs/).

## 💡 Contribute

We deeply value contributions from the open-source community to enhance Ultralytics projects. Your input helps drive innovation! Please review our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for detailed information on how to get involved. You can also share your feedback and ideas through our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A heartfelt thank you 🙏 to all our contributors for their dedication and support!

![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)

We look forward to your contributions!

## 📜 License

Ultralytics Docs are available under two licensing options to accommodate different usage scenarios:

- **AGPL-3.0 License**: Ideal for students, researchers, and enthusiasts involved in academic pursuits and open collaboration. See the [LICENSE](https://github.com/ultralytics/docs/blob/main/LICENSE) file for full details. This license promotes sharing improvements back with the community.
- **Enterprise License**: Designed for commercial applications, this license allows seamless integration of Ultralytics software and [AI models](https://docs.ultralytics.com/models/) into commercial products and services. Visit [Ultralytics Licensing](https://www.ultralytics.com/license) for more information on obtaining an Enterprise License.

## ✉️ Contact

For bug reports, feature requests, and other issues related to the documentation, please use [GitHub Issues](https://github.com/ultralytics/docs/issues). For discussions, questions, and community support, join the conversation with peers and the Ultralytics team on our [Discord server](https://discord.com/invite/ultralytics)!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
>>>>>>> 0e451281e978a7c5f581cc5d9bb3d0f62e0bc632
