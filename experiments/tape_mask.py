from collections import deque
from PIL import Image, ImageFilter

DARK = 60          # tape is ~20-45; bare metal 60-150
MIN_FRAC = 0.01    # a tape blob covers at least 1% of the frame
GROW = 4           # px; swallows the grey anti-aliased tape edge

def remove_tape(img):
    img = img.convert("RGB")
    w, h = img.size
    g = img.convert("L").load()
    seen = bytearray(w * h)
    tape = Image.new("L", (w, h), 0)
    tp = tape.load()
    border = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)] + \
             [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
    for sx, sy in border:
        if seen[sy * w + sx] or g[sx, sy] >= DARK:
            continue
        comp, q = [], deque([(sx, sy)])
        seen[sy * w + sx] = 1
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h and not seen[ny * w + nx] and g[nx, ny] < DARK:
                    seen[ny * w + nx] = 1
                    q.append((nx, ny))
        if len(comp) >= MIN_FRAC * w * h:          # big dark region touching the edge = tape
            for x, y in comp:
                tp[x, y] = 255
    if GROW:
        tape = tape.filter(ImageFilter.MaxFilter(2 * GROW + 1))
    # fill each tape pixel from the nearest non-tape pixel in the same row, so the table's
    # lighting gradient carries straight across instead of leaving a flat grey band
    px = img.load(); mp = tape.load()
    out = img.copy(); op = out.load()
    for y in range(h):
        row = [mp[x, y] for x in range(w)]
        if not any(row):
            continue
        good = [x for x in range(w) if not row[x]]
        if not good:
            continue
        left = [None] * w; right = [None] * w; last = None
        for x in range(w):
            if not row[x]: last = x
            left[x] = last
        last = None
        for x in range(w - 1, -1, -1):
            if not row[x]: last = x
            right[x] = last
        for x in range(w):
            if row[x]:
                a, b = left[x], right[x]
                if a is None: op[x, y] = px[b, y]
                elif b is None: op[x, y] = px[a, y]
                else:
                    t = (x - a) / (b - a)
                    pa, pb = px[a, y], px[b, y]
                    op[x, y] = tuple(int(pa[i] + (pb[i] - pa[i]) * t) for i in range(3))
    # the row-by-row copy leaves faint streaks from JPEG noise; soften only the filled area
    soft = out.filter(ImageFilter.GaussianBlur(6))
    out.paste(soft, mask=tape)
    return out, tape

if __name__ == "__main__":
    import sys
    a = Image.open(sys.argv[1]); b, m = remove_tape(a)
    W, H = a.size
    c = Image.new("RGB", (W * 2 + 10, H), (255, 255, 255)); c.paste(a.convert("RGB"), (0, 0)); c.paste(b, (W + 10, 0))
    c.save(sys.argv[2]); print("masked %.0f%%" % (100 * sum(1 for v in m.getdata() if v) / (W * H)))
