from pathlib import Path
STYLES_DIR = Path(__file__).parent
PUBLICATION_STYLE   = str(STYLES_DIR / "publication.mplstyle")
LATEX_PREAMBLE = str(STYLES_DIR / "latex_preamble.tex")
def use_publication_style() -> None:
    import matplotlib.pyplot as plt
    from figures.styles.font_config import apply_publication_fonts
    plt.style.use(PUBLICATION_STYLE)
    apply_publication_fonts()