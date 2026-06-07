from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


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


def _center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int = 8,
) -> None:
    line_sizes = [_text_size(draw, line, font) for line in lines]
    total_h = sum(h for _, h in line_sizes) + spacing * (len(lines) - 1)
    y = (box[1] + box[3] - total_h) // 2
    for line, (w, h) in zip(lines, line_sizes):
        x = (box[0] + box[2] - w) // 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + spacing


def _shadowed_round_rect(
    image: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int = 2,
    shadow: bool = True,
) -> None:
    draw = ImageDraw.Draw(image)
    if shadow:
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.rounded_rectangle(
            (box[0] + 8, box[1] + 10, box[2] + 8, box[3] + 10),
            radius=radius,
            fill=(35, 66, 105, 28),
        )
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(10)))
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 5,
    head: int = 18,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    points = [
        end,
        (
            int(end[0] - head * math.cos(angle - math.pi / 6)),
            int(end[1] - head * math.sin(angle - math.pi / 6)),
        ),
        (
            int(end[0] - head * math.cos(angle + math.pi / 6)),
            int(end[1] - head * math.sin(angle + math.pi / 6)),
        ),
    ]
    draw.polygon(points, fill=color)


def _chip(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    text_color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)
    w, h = _text_size(draw, text, font)
    draw.text(((box[0] + box[2] - w) // 2, (box[1] + box[3] - h) // 2 - 1), text, fill=text_color, font=font)


def render_route_a() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (1536, 1024), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    navy = (22, 59, 100)
    muted = (83, 95, 110)
    blue = (30, 102, 190)
    pale_blue = (235, 245, 255)
    green = (39, 139, 82)
    pale_green = (233, 250, 240)
    orange = (235, 124, 36)
    pale_orange = (255, 246, 235)
    purple = (103, 71, 196)
    pale_purple = (244, 241, 255)
    red = (206, 50, 54)
    pale_red = (255, 241, 241)
    border = (190, 205, 225)

    box_title_font = _font(BOLD, 42)
    box_body_font = _font(REGULAR, 28)
    chip_font = _font(BOLD, 24)
    small_font = _font(REGULAR, 22)
    tiny_bold = _font(BOLD, 20)

    fixed_box = (98, 200, 682, 690)
    loss_box = (854, 200, 1438, 690)
    _shadowed_round_rect(image, fixed_box, radius=30, fill=pale_green, outline=(117, 190, 150), width=3)
    _shadowed_round_rect(image, loss_box, radius=30, fill=pale_orange, outline=(242, 166, 88), width=3)

    _center_text(draw, (fixed_box[0], 246, fixed_box[2], 310), ["Fixed architecture"], box_title_font, green, spacing=0)
    _center_text(draw, (fixed_box[0], 338, fixed_box[2], 402), ["Chronos-T5"], box_title_font, navy, spacing=0)
    _center_text(draw, (fixed_box[0], 438, fixed_box[2], 512), ["same tokenizer", "same backbone"], box_body_font, muted, spacing=10)
    draw.rounded_rectangle((244, 572, 536, 636), radius=22, fill=(255, 255, 255, 230), outline=(117, 190, 150), width=2)
    _center_text(draw, (244, 572, 536, 636), ["unchanged"], chip_font, green, spacing=0)

    _center_text(draw, (loss_box[0], 246, loss_box[2], 310), ["Loss function"], box_title_font, orange, spacing=0)
    _center_text(draw, (loss_box[0], 338, loss_box[2], 392), ["changed per run"], box_body_font, muted, spacing=0)

    loss_items = [
        ("CE", blue, pale_blue, (133, 181, 228)),
        ("bin-index MSE", green, pale_green, (126, 198, 156)),
        ("bin-MSE + CE", green, pale_green, (126, 198, 156)),
        ("W1 / W2", purple, pale_purple, (174, 154, 232)),
        ("CRPS", purple, pale_purple, (174, 154, 232)),
        ("Ordinal CE", purple, pale_purple, (174, 154, 232)),
        ("Huber", green, pale_green, (126, 198, 156)),
    ]
    x_positions = [916, 1140]
    y_positions = [420, 493, 566]
    for idx, (label, color, fill, outline) in enumerate(loss_items[:6]):
        col = idx % 2
        row = idx // 2
        box = (x_positions[col], y_positions[row], x_positions[col] + 200, y_positions[row] + 56)
        _chip(draw, box, label, fill, outline, color, chip_font)
    _chip(draw, (1028, 632, 1228, 688), loss_items[-1][0], pale_green, (126, 198, 156), green, chip_font)

    draw.rounded_rectangle((1018, 760, 1274, 816), radius=20, fill=pale_red, outline=(235, 138, 142), width=2)
    _center_text(draw, (1018, 760, 1274, 816), ["only this changes"], tiny_bold, red, spacing=0)
    _arrow(draw, (1146, 696), (1146, 760), (235, 138, 142), width=4, head=16)

    _arrow(draw, (854, 440), (682, 440), navy, width=6, head=22)
    draw.rounded_rectangle((594, 360, 940, 420), radius=22, fill=(248, 251, 255), outline=border, width=2)
    _center_text(draw, (594, 360, 940, 420), ["LoRA fine-tune"], chip_font, navy, spacing=0)

    out = FIG_DIR / "route_a_chronos_loss_adaptation.png"
    image.convert("RGB").save(out, quality=95)
    return out


def main() -> None:
    render_route_a()


if __name__ == "__main__":
    main()
