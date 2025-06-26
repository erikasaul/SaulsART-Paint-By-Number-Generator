import streamlit as st, tempfile, pathlib, shutil
from saulsart_pbn_generator import generate_pbn

st.set_page_config(page_title="SaulsART Paint-By-Number", layout="wide")
st.title("🎨 SaulsART Paint-By-Number Generator")

# ---- sidebar widgets ----
n_colors     = st.sidebar.slider("Number of colours",        4, 30, 12)
min_area     = st.sidebar.slider("Minimum region area (px²)", 50, 1000, 200, 50)
number_color = st.sidebar.selectbox("Number colour", ["red", "black", "blue", "green"])
font_adjust  = st.sidebar.slider("Number font tweak",       -2, 2, 0)
make_svg     = st.sidebar.checkbox("Generate SVG template", True)

# ---- file upload ----
upl = st.file_uploader("Upload a photo (PNG / JPG)", type=["png", "jpg", "jpeg"])

if upl and st.button("Generate"):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        img_path = tmp / upl.name
        img_path.write_bytes(upl.read())

        out_dir = generate_pbn(
            img_path     = str(img_path),
            n_colors     = n_colors,
            min_area     = min_area,
            number_color = number_color,
            font_adjust  = font_adjust,
            make_svg     = make_svg,
            output_root  = "web_output"
        )

        st.success("Done ✔ Scroll down for previews")

        # --------------------------------------------------  show results
        st.subheader("Template")
        st.image(list(out_dir.glob("*_template.png"))[0])

        st.subheader("Colour guide")
        colour_png = (
            next(out_dir.glob("*_colours.png"), None)  # UK spelling
            or next(out_dir.glob("*_colors.png"),  None)  # US spelling
        )
        if colour_png:
            st.image(colour_png)
        else:
            st.warning("No colour-guide PNG was generated.")

        st.subheader("Reference")
        ref_png = next(out_dir.glob("*_reference.png"), None)
        if ref_png:
            st.image(ref_png)


        zip_path = shutil.make_archive(out_dir, "zip", out_dir)
        st.download_button("⭳ Download all assets (.zip)",
                           data=open(zip_path, "rb").read(),
                           file_name="paint_by_number_assets.zip")
