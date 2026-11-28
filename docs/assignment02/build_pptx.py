"""Fill Assignment_3D_motiondata.pptx template -> 4 content slides."""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

SRC, FIG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
BLUE = RGBColor(0x00, 0x6C, 0xB8)
DARK = RGBColor(0x33, 0x33, 0x33)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
ORANGE = RGBColor(0xE0, 0x7B, 0x39)
GREY = RGBColor(0x9A, 0x9A, 0x9A)
TINT = RGBColor(0xEE, 0xF5, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation(str(SRC))
slides = list(prs.slides)
content = slides[1:5]

# ---- drop cover (1) and closing (6) slides: max 4 slides
sldIdLst = prs.slides._sldIdLst
for s in (slides[0], slides[5]):
    for sldId in list(sldIdLst):
        if prs.part.related_part(sldId.rId) is s.part:
            prs.part.drop_rel(sldId.rId)
            sldIdLst.remove(sldId)


def shape(slide, name):
    return next(sh for sh in slide.shapes if sh.name == name)


def set_text(sh, text):
    """replace text but keep the first run's formatting"""
    p = sh.text_frame.paragraphs[0]
    r0 = p.runs[0]
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    r0.text = text


def remove(slide, *names):
    for n in names:
        el = shape(slide, n)._element
        el.getparent().remove(el)


def text(slide, x, y, w, h, paras, size=14, color=DARK, anchor=MSO_ANCHOR.TOP, name=None):
    """paras: list of str or list of (str, {bold, color, size}) runs; one entry per paragraph"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        p._p.get_or_add_pPr().set("latinLnBrk", "0"); p._p.get_or_add_pPr().set("eaLnBrk", "0")   # never break English words mid-word
        runs = [para] if isinstance(para, (str, tuple)) else para
        for run in runs:
            t, o = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", False)
            r.font.color.rgb = o.get("color", color)
            r._r.get_or_add_rPr().set("lang", "ko-KR")
    return tb


def card(slide, x, y, w, h, name):
    c = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    c.name = name
    c.adjustments[0] = 0.06
    c.fill.solid(); c.fill.fore_color.rgb = TINT
    c.line.fill.background()
    c.shadow.inherit = False
    return c


def picture(slide, path, x, y, w=None, h=None, name=None, alt=""):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    if name:
        pic.name = name
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic


def code_ref(slide, y, label):
    text(slide, 0.5, y, 12.3, 0.3, [[("코드  ", {"bold": True, "color": BLUE, "size": 12}),
                                       (label, {"color": MUTED, "size": 12})]], name="Code reference")


# ================================================================ slide 1: data acquisition
s = content[0]
set_text(shape(s, "TextBox 6"), "1. 데이터 수집")
remove(s, "TextBox 33", "TextBox 2", "TextBox 3")
picture(s, FIG / "fig1_acquisition.png", 0.5, 1.45, h=5.15, name="Figure: 2D and 3D landmarks",
        alt="One video frame with the BlazePose 2D skeleton (left) and the same frame's 3D world landmarks with the hip center at the origin (right)")
stats = [("44", "원본 영상"), ("5", "K-pop 스텝"), ("33", "관절 x (x, y, z)")]
for i, (big, small) in enumerate(stats):
    x = 8.15 + i * 1.62
    card(s, x, 1.55, 1.45, 1.3, f"Stat card {i + 1}")
    text(s, x, 1.68, 1.45, 0.6, [[(big, {"bold": True, "color": BLUE, "size": 30})]], name=f"Stat {i + 1}")
    s.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    text(s, x, 2.35, 1.45, 0.4, [[(small, {"color": MUTED, "size": 11})]], name=f"Stat label {i + 1}")
    s.shapes[-1].text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
text(s, 8.15, 3.1, 4.75, 3.6, [
    [("주제  ", {"bold": True, "color": BLUE}), ("3D 관절 좌표로 K-pop 기본 스텝 5가지(크랩, 펭귄, 힙합 투스텝, 찰스턴, 리버스 런닝맨)를 분류한다. 스텝은 발의 움직임으로 구분되므로 하체 10개 관절(주황)이 핵심이다.", {})],
    [("왜 골반 중심인가  ", {"bold": True, "color": BLUE}), ("MediaPipe world 좌표는 원점이 골반 중심으로 나온다. 서 있는 위치나 카메라 거리와 상관없이 자세(관절끼리의 상대 위치)만 비교할 수 있어 분류에 유리하다. 대신 몸 전체의 이동과 점프는 빠진다(문제 A).", {})],
    [("추출  ", {"bold": True, "color": BLUE}), ("BlazePose(model_complexity = 2)로 3D world 좌표(m) + 2D 이미지 좌표 + 가시성을 저장. 수집한 57개 중 발 가시성이 나쁜 13개를 빼고 44개 사용(모두 91% 이상의 프레임 검출)", {})],
], size=13, name="Acquisition details")
code_ref(s, 6.85, "scripts/p1_create_raw_keypoints.py --local")

# ================================================================ slide 2: problem identification
s = content[1]
set_text(shape(s, "TextBox 6"), "2. 문제 파악")
remove(s, "TextBox 33", "TextBox 3")
picture(s, FIG / "fig2_problems.png", 0.5, 1.4, w=12.3, name="Figure: three data problems",
        alt="A: 3D hip height stays at zero while the 2D hip moves. B: raw toe height jitters around a smooth trend. C: share of frames with low foot visibility per video")
cols = [
    ("A. 바닥과 이동 정보 없음", "발은 잘리거나 가려져 좌표가 자주 흔들리므로, 항상 안정적으로 잡히는 골반을 기준점(원점)으로 쓴다(44개 모두 |골반| ≤ 5mm). 대신 바닥 접촉, 점프, 옆 이동(크랩)은 3D에서 보이지 않는다."),
    ("B. 프레임 간 떨림", "관절이 프레임마다 흔들린다(raw 하체 떨림: 프레임당 다리 길이의 0.018배). 스텝 지표를 좌우하는 발에서 특히 심하다."),
    ("C. 발 좌표 추출의 어려움", "발이 화면 아래로 잘리거나 밖으로 나가고, 바지나 다른 발에 가려지면 발 좌표의 신뢰도(가시성)가 떨어진다. 전체 프레임의 14.8%가 이런 경우다. 이런 프레임이 절반을 넘던 13개 영상은 수집 단계에서 뺐다."),
]
for i, (head, body) in enumerate(cols):
    x = 0.5 + i * 4.2
    card(s, x, 4.95, 3.9, 1.85, f"Problem card {i + 1}")
    text(s, x + 0.2, 5.05, 3.5, 1.7, [[(head, {"bold": True, "color": BLUE, "size": 14})],
                                      [(body, {"size": 12})]], name=f"Problem {i + 1}")
code_ref(s, 6.95, "scripts/p1_3_analyze_postprocessing.py   (지표 단위: 다리 길이 = 골반 중심에서 발목 중심까지)")

# ================================================================ slide 3: post-processing
s = content[2]
set_text(shape(s, "TextBox 6"), "3. 후처리")
remove(s, "TextBox 33", "TextBox 3")
card(s, 0.5, 1.45, 12.35, 1.95, "Method card")
steps = [("① 높이 보정 → A  ", "매 프레임 더 낮은 발끝을 바닥(y = 0)에 맞춘다:  ground(t) = min(y_왼발끝, y_오른발끝),  y ← max(y − ground, 0)"),
         ("② 빈 구간 보간 → C  ", "포즈가 빠지거나 발 좌표 신뢰도가 낮은 구간 중 0.33초 이하를 앞뒤 값으로 채운다"),
         ("③ 몸 전체 위치 복원 → A  ", "①은 점프까지 바닥에 붙이므로, 2D 영상의 골반 위치를 m 단위로 바꿔 더하고 바닥은 영상 전체에서 한 번만 정한다"),
         ("④ Savitzky-Golay 평활화 → B  ", "창 9프레임(0.3초), 3차 다항식. 스텝의 최고점은 유지된다")]
text(s, 0.75, 1.55, 11.9, 1.8, [[("문제 해결을 위한 후처리", {"bold": True, "color": BLUE, "size": 16}),
                                 ("   ①은 제공 코드, ②~④는 추가 구현", {"color": MUTED, "size": 12})]] +
     [[(h, {"bold": True, "color": BLUE, "size": 12}), (b, {"size": 12})] for h, b in steps], name="Method steps")
picture(s, FIG / "fig3_postprocessing.png", 0.5, 3.5, w=12.3, name="Figure: improved post-processing",
        alt="A: lowest-foot height is pinned to zero by the provided correction but follows the 2D reference after the improved method. B: toe height jitter before and after smoothing. C: short low-visibility stretches interpolated, long ones left as detected")
code_ref(s, 6.95, "①  scripts/p1_2_post_processing.py      ②~④  scripts/p1_3_analyze_postprocessing.py (improved_postprocess)")

# ================================================================ slide 4: results & analysis
s = content[3]
set_text(shape(s, "TextBox 6"), "4. 결과 및 분석")
remove(s, "TextBox 33", "TextBox 3")
rows = [("지표 (44개 영상 중앙값)", "Raw", "① 높이 보정", "① + ④", "②+③+④"),
        ("양발이 모두 뜬 높이, p95", "0.134", "0.000", "0.003", "0.127"),
        ("2D 점프 크기와의 상관", "0.27", "-", "0.26", "0.59"),
        ("발 드는 높이, p95", "0.280", "0.229", "0.221", "0.291"),
        ("하체 떨림", "0.0179", "0.0186", "0.0057", "0.0060"),
        ("발목 jerk RMS", "1933", "2017", "364", "374")]
tbl = s.shapes.add_table(len(rows), 5, Inches(0.5), Inches(1.5), Inches(7.0), Inches(2.6))
tbl.name = "Metrics table"
t = tbl.table
for j, w in enumerate([2.55, 0.9, 1.25, 1.05, 1.25]):
    t.columns[j].width = Inches(w)
for i, row in enumerate(rows):
    for j, v in enumerate(row):
        c = t.cell(i, j)
        c.fill.solid(); c.fill.fore_color.rgb = BLUE if i == 0 else (TINT if (j == 4 or i % 2 == 0) else WHITE)
        c.margin_left = c.margin_right = Inches(0.08)
        p = c.text_frame.paragraphs[0]
        p._p.get_or_add_pPr().set("latinLnBrk", "0"); p._p.get_or_add_pPr().set("eaLnBrk", "0")
        p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
        r = p.add_run(); r.text = v
        r.font.size = Pt(12); r.font.bold = (i == 0 or j == 4)
        r.font.color.rgb = WHITE if i == 0 else DARK
        r._r.get_or_add_rPr().set("lang", "ko-KR")
text(s, 0.5, 4.2, 7.0, 0.3, [[("단위: 다리 길이 (jerk: 다리 길이 / s³). 상관: 영상 44개에 걸친 피어슨 상관", {"color": MUTED, "size": 10})]], name="Table note")

cd = CategoryChartData()
cd.categories = ["양발이 뜬 높이", "떨림", "발목 jerk"]
cd.add_series("① 높이 보정", (0, 104.2, 104.4))
cd.add_series("① + ④", (2.1, 31.8, 18.9))
cd.add_series("②+③+④", (94.7, 33.3, 19.3))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(7.75), Inches(1.45), Inches(5.1), Inches(3.05), cd)
gf.name = "Chart: percent of raw"
ch = gf.chart
ch.has_title = True
ch.chart_title.text_frame.text = "Raw 대비 비율 (%, Raw = 100)"
tr = ch.chart_title.text_frame.paragraphs[0].runs[0]
tr.font.size = Pt(13); tr.font.bold = True; tr.font.color.rgb = DARK
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11); ch.legend.font.color.rgb = DARK
va = ch.value_axis
va.maximum_scale = 120; va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xE3, 0xE3)
va.tick_labels.font.size = Pt(10); va.tick_labels.font.color.rgb = MUTED
va.format.line.fill.background()
ca = ch.category_axis
ca.tick_labels.font.size = Pt(11); ca.tick_labels.font.color.rgb = DARK
for ser, col in zip(ch.plots[0].series, [ORANGE, GREY, BLUE]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
plot = ch.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format = '0'; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(10); dl.font.color.rgb = DARK

findings = [
    ("① 높이 보정의 한계: ", "매 프레임 더 낮은 발을 y = 0에 붙여 점프가 지워지고('양발이 뜬 높이' 0.134 → 0), 흔들리는 발끝을 바닥으로 삼아 떨림이 오히려 4% 늘어난다."),
    ("②~④로 해결: ", "바닥을 영상 전체에서 한 번만 정하고 2D의 몸 위치를 더해 점프를 살렸다(0.127, 2D 점프와의 상관 0.27 → 0.59). 떨림 −67%, jerk −81%, 발 드는 높이는 유지."),
    ("남은 한계: ", "발 좌표 신뢰도가 낮은 관절·프레임 3,435개 중 짧은 구간 759개(22%)만 ②로 채웠다. 긴 구간은 검출된 좌표를 그대로 쓴다. ③은 고정 카메라와 일정한 촬영 거리를 가정한다."),
]
card(s, 0.5, 4.65, 12.35, 2.05, "Findings card")
text(s, 0.75, 4.78, 11.9, 1.85, [[(h, {"bold": True, "color": BLUE, "size": 13}), (b, {"size": 13})] for h, b in findings],
     name="Findings")
code_ref(s, 6.85, "scripts/p1_3_analyze_postprocessing.py  ->  data/output/all_metrics.csv")

prs.save(str(OUT))
print("saved", OUT)
