"""
Find color schemes with optimally distinct colors using Delta E (CIE2000).

Steps to generate optimal color schemes:

  1. Compile list of potential colors.
  2. Calculate Delta E between all colors.
  3. Compile list of potential schemes with 6 colors each.
  4. Discard schemes if they contain hues that are too similar or if their
     Delta E of any two colors is too low.
  5. Output remaining color schemes.

"""

from datetime import datetime
from itertools import combinations
from itertools import zip_longest
from math import factorial
from multiprocessing import cpu_count
from multiprocessing import Lock
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Value

from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import HSLColor
from colormath.color_objects import LabColor
from colormath.color_objects import sRGBColor

# Output formats: 'hex', 'hsl', 'rgb' (0–1) or 'rgb_upscaled' (0–255)
color_codes = ['hex']
# Bright or dark background
bright = True
# Discard scheme if any hue combination is less than this many degrees apart
min_hue_diff = 30
# Spacing of hues in degrees, smaller spacing results in more potential schemes
# (steps of 1° yield 2899305949260 schemes)
# Factors of 360:
# 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60
hue_step = 10
# Enable to use more RAM for better performance, disable to process larger
# numbers of schemes that will not fit into your RAM
load_schemes_into_ram = False
# Output schemes as soon as they are found
output_schemes_early = False
# Parameters depending on bright or dark background
if bright:
    # One color can be adjusted to be used as a highlight color (e.g. when
    # searching in a man page)
    # Colors: 'red', 'yellow', 'green', 'cyan', 'blue', 'magenta'
    highlight_color = 'yellow'
    # Minimum Delta E between the highlight color and the background
    min_delta_background_highlight = 30
    # Minimum Delta E between all other colors and the background
    min_delta_background_color = 55
    # Luminance is adjusted this much each step until the required delta to the
    # background is achieved
    lum_adjust_highlight = -.01
    lum_adjust_color = -.01
    # Fixed colors, use for example:
    #   sRGBColor.new_from_rgb_hex('#000000')
    #   sRGBColor(128, 0, 255, is_upscaled=True)
    #   sRGBColor(.5, 0, 1)
    #   HSLColor(0, 0, .8)
    # Black
    color0 = sRGBColor.new_from_rgb_hex('#000000')
    color8 = sRGBColor.new_from_rgb_hex('#000000')
    # White
    color7 = sRGBColor.new_from_rgb_hex('#cccccc')
    color15 = sRGBColor.new_from_rgb_hex('#ffffff')
    # Background
    background = color15
    # Exclude hues
    excluded_hues = []
else:
    # One color can be adjusted to be used as a highlight color (e.g. when
    # searching in a man page)
    # Colors: 'red', 'yellow', 'green', 'cyan', 'blue', 'magenta'
    highlight_color = 'blue'
    # Minimum Delta E between the highlight color and the background
    min_delta_background_highlight = 30
    # Minimum Delta E between all other colors and the background
    min_delta_background_color = 60
    # Luminance is adjusted this much each step until the required delta to the
    # background is achieved
    lum_adjust_highlight = .01
    lum_adjust_color = .01
    # Fixed colors, use for example:
    #   sRGBColor.new_from_rgb_hex('#000000')
    #   sRGBColor(128, 0, 255, is_upscaled=True)
    #   sRGBColor(.5, 0, 1)
    #   HSLColor(0, 0, .8)
    # Black
    color0 = sRGBColor.new_from_rgb_hex('#000000')
    color8 = sRGBColor.new_from_rgb_hex('#000000')
    # White
    color7 = sRGBColor.new_from_rgb_hex('#555555')
    color15 = sRGBColor.new_from_rgb_hex('#ffffff')
    # Background
    background = color8
    # Exclude hues
    excluded_hues = []

# Convert colors
color0 = convert_color(color0, LabColor)
color8 = convert_color(color8, LabColor)
color7 = convert_color(color7, LabColor)
color15 = convert_color(color15, LabColor)
background = convert_color(background, LabColor)
highlight = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta'].index(
    highlight_color)


def convert_hue(hue, adjust=lum_adjust_color,
                min_delta=min_delta_background_color):
    """Create a color from a hue and adjust its luminance until the Delta E
    between the color and the background is sufficient.
    """
    color_hsl = HSLColor(hue, 1, .5)
    color_lab = convert_color(color_hsl, LabColor)
    while True:
        if delta_e_cie2000(color_lab, background) < min_delta:
            color_hsl.hsl_l += adjust
            color_lab = convert_color(color_hsl, LabColor)
        else:
            return [hue, color_lab]


def calculate_delta(colors):
    """Calculate Delta E between two colors."""
    # Structure of dictionary: dict([(hue_1, hue_2): delta_e, ...])
    global deltas
    deltas[(colors[0][0], colors[1][0])] = \
        delta_e_cie2000(colors[0][1], colors[1][1])


def check_scheme(scheme):
    """Check a scheme for similar hues and low Delta E values. Discard if
    insufficient.
    """
    # Due to lazy evaluation with grouper(), scheme can be None
    if not scheme:
        return
    global lock
    # Show progress
    global processed
    with lock:
        processed.value += 1
        processed_schemes = processed.value
    if processed_schemes % 100000 == 0:
        elapsed = datetime.utcnow() - start_time
        progress = processed_schemes / total_schemes
        print('== Progress: {}/{} {}%'.format(
            processed_schemes, total_schemes,
            int(round(progress * 100))))
        print('   Elapsed time: {}'.format(str(elapsed).split('.')[0]))
        print('   Time left: {}'.format(
            str(elapsed * (1 / progress) - elapsed).split('.')[0]))
    # Check hues
    min_hue_diff_scheme = \
        min([scheme[i + 1][0] - scheme[i][0] for i in range(5)] +
            [scheme[0][0] + 360 - scheme[-1][0]])
    if min_hue_diff_scheme < min_hue_diff:
        return
    # Check Delta E values
    min_delta = 100
    global current_min_delta
    for x, y in combinations(scheme, 2):
        delta = deltas[(x[0], y[0])]
        if delta < current_min_delta.value:
            return
        min_delta = min(min_delta, delta)
    new_best = False
    notify = False
    with lock:
        if current_min_delta.value < min_delta:
            new_best = True
            if int(current_min_delta.value) != int(min_delta):
                notify = True
            current_min_delta.value = min_delta
    if notify:
        print('== New minimum Delta E: {}'.format(min_delta))
    if new_best and output_schemes_early:
        output_scheme(finalize_scheme(
            [min_delta, min_hue_diff_scheme, scheme]))
    return [min_delta, min_hue_diff_scheme, scheme]


def finalize_scheme(scheme):
    """Finalize a scheme by brightening its highlight color until it reaches
    the chosen Delta E to the background and ordering the colors of the scheme.
    """
    scheme[2] = list(scheme[2])
    hues = [hue for hue, color in scheme[2]]
    # Make sure the first hue is closest to red
    if 360 - max(hues) < min(hues):
        scheme[2].insert(0, scheme[2].pop())
        hues.insert(0, hues.pop())
    # Adjust luminance of highlight color
    color = convert_hue(hues[highlight], lum_adjust_highlight,
                        min_delta_background_highlight)[1]
    # Add adjusted color
    scheme[2][highlight] = (hues[highlight], color)
    return scheme


def color_to_str(color):
    """Return a color’s hex, sRGB or HSL representations."""
    s = []
    if 'hex' in color_codes:
        s.append(str(convert_color(color, sRGBColor).get_rgb_hex()))
    if 'rgb_upscaled' in color_codes:
        s.append('rgb{}'.format(
            str(convert_color(color, sRGBColor).get_upscaled_value_tuple())))
    if 'rgb' in color_codes:
        s.append('rgb{}'.format(
            str(convert_color(color, sRGBColor).get_value_tuple())))
    if 'hsl' in color_codes:
        s.append('hsl{}'.format(
            str(convert_color(color, HSLColor).get_value_tuple())))
    return ', '.join(s)


def output_scheme(scheme):
    """Output a color scheme."""
    hues = [color[0] for color in scheme[2]]
    if scheme[1] - min_hue_diff < hue_step:
        print('# Warning: Scheme borders minimal hue difference')
    print('# Minimum Delta E: {}'.format(scheme[0]))
    print('# Minimum hue difference: {}'.format(scheme[1]))
    print('# Hues: {}'.format(' '.join(map(str, hues))))
    print('# black')
    print('color0  = {}'.format(color_to_str(color0)))
    print('color8  = {}'.format(color_to_str(color8)))
    print('# red')
    print('color1  = {}'.format(color_to_str(scheme[2][0][1])))
    print('color9  = {}'.format(color_to_str(scheme[2][0][1])))
    print('# green')
    print('color2  = {}'.format(color_to_str(scheme[2][2][1])))
    print('color10 = {}'.format(color_to_str(scheme[2][2][1])))
    print('# yellow')
    print('color3  = {}'.format(color_to_str(scheme[2][1][1])))
    print('color11 = {}'.format(color_to_str(scheme[2][1][1])))
    print('# blue')
    print('color4  = {}'.format(color_to_str(scheme[2][4][1])))
    print('color12 = {}'.format(color_to_str(scheme[2][4][1])))
    print('# magenta')
    print('color5  = {}'.format(color_to_str(scheme[2][5][1])))
    print('color13 = {}'.format(color_to_str(scheme[2][5][1])))
    print('# cyan')
    print('color6  = {}'.format(color_to_str(scheme[2][3][1])))
    print('color14 = {}'.format(color_to_str(scheme[2][3][1])))
    print('# white')
    print('color7  = {}'.format(color_to_str(color7)))
    print('color15 = {}'.format(color_to_str(color15)))


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks."""
    # Source: https://docs.python.org/3/library/itertools.html#recipes
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


if __name__ == "__main__":
    start_time = datetime.utcnow()
    processes = cpu_count()
    manager = Manager()

    # Compile list of hues
    hues = range(0, 360, hue_step)
    hues = [hue for hue in hues if hue not in excluded_hues]

    # Compile list of potential colors as (hue, LabColor) pairs
    colors = manager.list()
    with Pool(processes) as pool:
        colors = pool.map(convert_hue, hues)
        pool.close()
        pool.join()
    colors.sort()
    print('Colors prepared')

    # Calculate Delta E between all colors
    deltas = manager.dict()
    with Pool(processes) as pool:
        pool.map(calculate_delta, combinations(colors, 2))
        pool.close()
        pool.join()
    # Making a dictionary now from multiprocessing.managers.DictProxy will
    # increase subsequent performance a lot
    deltas = dict(deltas)
    print('Color deltas calculated')

    # Compile list of potential schemes with 6 colors each
    schemes = combinations(colors, 6)
    total_schemes = int(
        factorial(len(colors)) / (factorial(6) * factorial(len(colors) - 6)))
    print('Total schemes: {}'.format(total_schemes))

    # Discard schemes if they contain hues that are too similar or if their
    # Delta E of any two colors is too low
    current_min_delta = Value('f', 0)
    processed = Value('i', 0)
    lock = Lock()
    schemes_checked = []
    # Process a known reference scheme with high Delta E values, so worse
    # schemes can be discarded faster in the next step
    with Pool(1) as pool:
        if bright:
            ref_hues = [5, 45, 100, 180, 220, 310]
        else:
            ref_hues = [5, 45, 100, 180, 220, 310]
        ref_hues = [min(hues, key=lambda x: abs(x - hue)) for hue in ref_hues]
        ref_scheme = [color for color in colors if color[0] in ref_hues]
        schemes_checked.extend(pool.map(check_scheme, [ref_scheme]))
        pool.close()
        pool.join()
    # Process all other schemes
    with Pool(processes) as pool:
        if load_schemes_into_ram:
            schemes_checked.extend(pool.map(check_scheme, schemes))
        else:
            # Chunks of 500000 have been fastest in my testing, but it probably
            # depends on the number of schemes
            # (Note that there is some overhead for the last chunk because it
            # gets filled with None values to reach the chunk size)
            for chunk in grouper(schemes, 500000):
                schemes_checked.extend(pool.map(check_scheme, chunk))
        pool.close()
        pool.join()

    # Remove None values and sort schemes by their minimum Delta E
    schemes_checked = sorted([scheme for scheme in schemes_checked if scheme],
                             key=lambda x: x[0], reverse=True)

    elapsed = datetime.utcnow() - start_time
    print('Total time: {}'.format(str(elapsed).split('.')[0]))

    # Output optimal color schemes
    print('=====================')
    print('Optimal color schemes')
    print('=====================')
    last_delta = None
    for scheme in schemes_checked:
        if last_delta and scheme[0] < last_delta:
            break
        output_scheme(finalize_scheme(scheme))
        last_delta = scheme[0]
