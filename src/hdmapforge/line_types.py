"""JD output labels and the client's confirmed coarse source mapping."""

JD_TYPES = {
    0: "未知",
    1: "单虚线",
    2: "双虚线",
    3: "单实线",
    4: "双实线",
    5: "左虚右实",
    6: "左实右虚",
    7: "马路牙",
    8: "防护栏",
    9: "虚拟线",
    10: "绿化带",
    11: "道路边界线",
}
SOURCE_FAMILIES = {1: frozenset({1, 2}), 2: frozenset({3, 4}), 3: frozenset({7}), 4: frozenset({8})}


def allowed_types(source_type=None):
    if source_type is None:
        return frozenset(JD_TYPES)
    if source_type not in SOURCE_FAMILIES:
        raise ValueError("Unknown source TYPE")
    return SOURCE_FAMILIES[source_type] | {0}
