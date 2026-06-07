from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"


def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


BOLD = "DejaVuSans-Bold.ttf"
REGULAR = "DejaVuSans.ttf"


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _rounded_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_color: tuple[int, int, int],
    outline: tuple[int, int, int],
    font: ImageFont.ImageFont,
    pad_x: int = 14,
    pad_y: int = 7,
    fill: tuple[int, int, int, int] = (255, 255, 255, 232),
) -> tuple[int, int, int, int]:
    x, y = xy
    width, height = _text_size(draw, text, font)
    box = (x, y, x + width + pad_x * 2, y + height + pad_y * 2)
    draw.rounded_rectangle(box, radius=11, fill=fill, outline=outline, width=1)
    draw.text((x + pad_x, y + pad_y - 1), text, fill=text_color, font=font)
    return box


def render_classification() -> Path:
    image = Image.open(FIG_DIR / "challenge_classification_ignores_distance_raw.png").convert("RGBA")
    draw = ImageDraw.Draw(image)

    navy = (22, 59, 100)
    muted = (84, 95, 109)
    red = (220, 42, 36)
    orange = (246, 126, 25)
    green = (43, 139, 80)

    title_font = _font(BOLD, 50)
    subtitle_font = _font(REGULAR, 29)
    label_font = _font(BOLD, 25)
    small_label_font = _font(BOLD, 24)

    draw.rounded_rectangle(
        (36, 30, 1500, 155),
        radius=31,
        fill=(255, 255, 255, 238),
        outline=(190, 205, 225),
        width=1,
    )
    draw.text((72, 52), "Classification ignores distance", fill=navy, font=title_font)
    draw.text(
        (73, 116),
        "Plain CE: nearby and far-away wrong bins are both wrong classes.",
        fill=muted,
        font=subtitle_font,
    )

    same_box = _rounded_label(
        draw,
        (650, 198),
        "same CE penalty",
        text_color=red,
        outline=(255, 132, 125),
        font=label_font,
    )

    near_box = _rounded_label(
        draw,
        (260, 232),
        "near wrong",
        text_color=orange,
        outline=(255, 168, 85),
        font=small_label_font,
    )
    far_box = _rounded_label(
        draw,
        (1190, 232),
        "far wrong",
        text_color=orange,
        outline=(255, 168, 85),
        font=small_label_font,
    )

    # Short callout strokes keep the wrong-bin labels readable without spanning the full panel.
    draw.line((near_box[2] + 10, 254, same_box[0] - 15, 224), fill=red, width=3)
    draw.line((same_box[2] + 15, 224, far_box[0] - 18, 254), fill=red, width=3)

    _rounded_label(
        draw,
        (585, 304),
        "correct bin",
        text_color=green,
        outline=(91, 185, 130),
        font=label_font,
    )

    out = FIG_DIR / "challenge_classification_ignores_distance.png"
    image.save(out)
    return out


def render_two_panel() -> Path:
    left = Image.open(FIG_DIR / "challenge_classification_ignores_distance.png").convert("RGB")
    right = Image.open(FIG_DIR / "challenge_regression_conservative.png").convert("RGB")
    left = left.resize((960, 640), Image.Resampling.LANCZOS)
    right = right.resize((960, 640), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (2000, 694), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((8, 8, 1992, 686), radius=30, outline=(190, 205, 225), width=2)
    canvas.paste(left, (40, 30))
    canvas.paste(right, (1040, 30))

    out = FIG_DIR / "challenge_diagrams_two_panel.png"
    canvas.save(out)
    return out


def main() -> None:
    render_classification()
    render_two_panel()


if __name__ == "__main__":
    main()
