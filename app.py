"""PrivacySweep - Streamlit dashboard.

A privacy-awareness prototype that screens user-provided listing descriptions
for Indian mobile-number patterns and public-exposure keywords, combined with a
TF-IDF + Logistic Regression text classifier.
"""

import io
import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PrivacySweep", page_icon="🛡️", layout="wide")

MODEL_PATH = Path(__file__).parent / "privacy_model.joblib"

# ----------------------------------------------------------------------
# Detection settings (same rules as the notebook)
# ----------------------------------------------------------------------
MOBILE_PATTERN = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")

EXPOSURE_KEYWORDS = [
    "public directory",
    "public listing",
    "public website",
    "public profile",
    "visible publicly",
    "listed online",
    "exposed online",
    "available online",
    "search results",
    "business directory",
]

NEGATIVE_EXPOSURE_PHRASES = [
    "no public listing",
    "no public website exposure",
    "no public exposure",
    "not publicly listed",
    "not exposed publicly",
    "not available online",
    "no exposure online",
]

SAMPLES = {
    "High Review Priority example": (
        "Contact details are available in this public business directory. "
        "Call 9876543210 for further information."
    ),
    "Needs Verification example": (
        "Spacious 2BHK flat for rent near the metro station. Call 9123456789."
    ),
    "No Review example": (
        "Spacious 2BHK flat for rent near the metro station. "
        "Contact the owner through the app."
    ),
}


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = joblib.load(MODEL_PATH)
    # Compatibility fix: models saved with newer scikit-learn lack `multi_class`
    classifier = model.steps[-1][1]
    if not hasattr(classifier, "multi_class"):
        classifier.multi_class = "auto"
    return model


def privacy_predict(model, description):
    text = description.lower()

    mobile_numbers = MOBILE_PATTERN.findall(description)

    negative = any(p in text for p in NEGATIVE_EXPOSURE_PHRASES)
    exposure_found = any(k in text for k in EXPOSURE_KEYWORDS) and not negative

    prediction = model.predict([description])[0]
    probabilities = model.predict_proba([description])[0]
    classes = list(model.classes_)
    review_p = float(probabilities[classes.index("Review")]) if "Review" in classes else 0.0
    no_review_p = (
        float(probabilities[classes.index("No Review")]) if "No Review" in classes else 0.0
    )

    if mobile_numbers and exposure_found:
        assessment, context, high = "High Review Priority", "Public exposure detected", True
    elif mobile_numbers:
        assessment, context, high = "Needs Verification", "Not detected", False
    else:
        assessment, context, high = str(prediction), "Not detected", False

    return {
        "assessment": assessment,
        "mobile_count": len(mobile_numbers),
        "exposure_context": context,
        "prediction": str(prediction),
        "review_probability": review_p,
        "no_review_probability": no_review_p,
        "high_priority": high,
    }


def build_removal_draft(listing_url="", your_name=""):
    """A plain template. It deliberately does not repeat the phone number."""
    return (
        "Subject: Request to remove personal contact information from a listing\n\n"
        "Hello,\n\n"
        "I am writing to request the removal of a personal mobile number that appears "
        f"in the public listing at {listing_url or '[LISTING URL]'}.\n\n"
        "The number is shown in the listing description and is not intended for public "
        "use. Please remove it, or edit the listing so it no longer displays this "
        "information, and confirm once this is done.\n\n"
        f"Thank you,\n{your_name or '[YOUR NAME]'}"
    )


def read_csv_bytes(raw_bytes):
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            decoded = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            decoded = None
    if decoded is None:
        raise ValueError("Could not decode the CSV file.")
    # Default comma-separated read first. Only auto-detect the delimiter when the
    # header suggests another separator (semicolon, tab or pipe).
    df = pd.read_csv(io.StringIO(decoded))
    if df.shape[1] == 1 and any(d in str(df.columns[0]) for d in (";", "\t", "|")):
        try:
            df = pd.read_csv(io.StringIO(decoded), sep=None, engine="python")
        except Exception:
            pass
    return df


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None
if "drafts" not in st.session_state:
    st.session_state.drafts = 0
if "listing_text" not in st.session_state:
    st.session_state.listing_text = ""

model = load_model()

ICONS = {"High Review Priority": "🔴", "Needs Verification": "🟡", "No Review": "🟢"}

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🛡️ PrivacySweep")
st.caption(
    "Privacy Intelligence Dashboard · Detect exposure · Review risk · "
    "Prepare user-submitted removal requests"
)

tab_single, tab_batch, tab_history, tab_about = st.tabs(
    ["🔍 Analyze a listing", "📂 CSV batch analysis", "🕘 Analysis history", "ℹ️ About"]
)

# ----------------------------------------------------------------------
# Tab 1: single listing
# ----------------------------------------------------------------------
with tab_single:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Analyze a listing")

        sample_choice = st.selectbox(
            "Try a sample (optional)", ["-- choose --"] + list(SAMPLES.keys())
        )
        if sample_choice != "-- choose --":
            st.session_state.listing_text = SAMPLES[sample_choice]

        description = st.text_area(
            "Enter the public listing description",
            value=st.session_state.listing_text,
            height=150,
            placeholder="Paste a listing description here. Use fake numbers for demos.",
        )

        if st.button("🔍 Analyze listing", type="primary"):
            if not description.strip():
                st.warning("Enter a listing description first.")
            else:
                try:
                    result = privacy_predict(model, description.strip())
                    st.session_state.last = result
                    st.session_state.history.append(
                        {
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Description": description.strip()[:90],
                            "Assessment": result["assessment"],
                            "Mobile Numbers": result["mobile_count"],
                            "Review Probability (%)": round(
                                result["review_probability"] * 100, 1
                            ),
                        }
                    )
                    if result["high_priority"]:
                        st.session_state.drafts += 1
                except Exception as exc:  # show a clear message instead of crashing
                    st.error(f"Model error: {type(exc).__name__}: {exc}")

    with right:
        st.subheader("Removal request assistant")
        result = st.session_state.last
        if result is None:
            st.info("Analyze a listing to see results here.")
        elif not result["high_priority"]:
            st.info("No draft needed. This listing did not receive High Review Priority.")
        else:
            st.caption("Optional: fill these in, then edit the draft below.")
            url = st.text_input("Listing URL", placeholder="https://...")
            name = st.text_input("Your name")
            st.text_area(
                "Editable draft",
                value=build_removal_draft(url, name),
                height=280,
                key=f"draft_{url}_{name}",
            )
            st.caption("PrivacySweep does not send this request. You submit it yourself.")

    result = st.session_state.last
    if result is not None:
        st.divider()
        st.subheader("Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Final assessment",
            f"{ICONS.get(result['assessment'], '')} {result['assessment']}",
        )
        c2.metric("Mobile patterns found", result["mobile_count"])
        c3.metric("Public exposure", result["exposure_context"])
        c4.metric("ML prediction", result["prediction"])

        st.write(f"**Review probability:** {result['review_probability']:.1%}")
        st.progress(min(max(result["review_probability"], 0.0), 1.0))
        st.write(f"**No-review probability:** {result['no_review_probability']:.1%}")
        st.progress(min(max(result["no_review_probability"], 0.0), 1.0))
        st.caption(
            "The final assessment comes from the rules (number pattern + exposure keywords). "
            "The percentages are the ML classifier's scores. They are not proof that a "
            "number is publicly exposed."
        )

# ----------------------------------------------------------------------
# Tab 2: batch
# ----------------------------------------------------------------------
with tab_batch:
    st.subheader("CSV batch analysis")
    st.write("Upload a CSV, choose the text column, and analyze many descriptions at once.")

    sample_csv = pd.DataFrame(
        {"description": list(SAMPLES.values())
         + ["Business opening hours displayed on the website."]}
    ).to_csv(index=False)
    st.download_button(
        "⬇️ Download a sample CSV", sample_csv, "sample_listings.csv", "text/csv"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = read_csv_bytes(uploaded.getvalue())
            if df.empty:
                raise ValueError("The CSV file contains no records.")
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            df = None

        if df is not None:
            preferred = next(
                (c for c in df.columns
                 if str(c).strip().lower() in {"description", "listing_description", "text"}),
                df.columns[0],
            )
            column = st.selectbox(
                "Text column", list(df.columns), index=list(df.columns).index(preferred)
            )
            st.caption(f"Loaded {len(df)} records.")

            if st.button("Analyze CSV", type="primary"):
                rows = []
                with st.spinner("Analyzing..."):
                    for value in df[column]:
                        text = "" if pd.isna(value) else str(value).strip()
                        if not text:
                            rows.append(
                                {
                                    "Assessment": "Needs Verification",
                                    "Mobile Numbers Found": 0,
                                    "Exposure Context": "Empty description",
                                    "ML Prediction": "Not analyzed",
                                    "Review Probability (%)": None,
                                    "High Priority": False,
                                }
                            )
                        else:
                            r = privacy_predict(model, text)
                            rows.append(
                                {
                                    "Assessment": r["assessment"],
                                    "Mobile Numbers Found": r["mobile_count"],
                                    "Exposure Context": r["exposure_context"],
                                    "ML Prediction": r["prediction"],
                                    "Review Probability (%)": round(
                                        r["review_probability"] * 100, 2
                                    ),
                                    "High Priority": r["high_priority"],
                                }
                            )
                out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total analyzed", len(out))
                m2.metric("🔴 High priority", int((out["Assessment"] == "High Review Priority").sum()))
                m3.metric("🟡 Needs verification", int((out["Assessment"] == "Needs Verification").sum()))
                m4.metric("🟢 No review", int((out["Assessment"] == "No Review").sum()))

                st.dataframe(out, width="stretch")
                st.download_button(
                    "⬇️ Download results",
                    out.to_csv(index=False),
                    "privacysweep_results.csv",
                    "text/csv",
                )

# ----------------------------------------------------------------------
# Tab 3: history
# ----------------------------------------------------------------------
with tab_history:
    st.subheader("Analysis history")
    history = st.session_state.history
    if not history:
        st.info("No analyses yet in this session.")
    else:
        hist_df = pd.DataFrame(history)
        h1, h2, h3 = st.columns(3)
        h1.metric("Listings analyzed", len(hist_df))
        h2.metric("Removal drafts generated", st.session_state.drafts)
        h3.metric(
            "High priority",
            int((hist_df["Assessment"] == "High Review Priority").sum()),
        )
        st.dataframe(hist_df, width="stretch")
        st.caption("Review probability per analysis")
        st.bar_chart(hist_df["Review Probability (%)"])
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.last = None
            st.session_state.drafts = 0
            st.rerun()

# ----------------------------------------------------------------------
# Tab 4: about
# ----------------------------------------------------------------------
with tab_about:
    st.subheader("About PrivacySweep")
    st.markdown(
        """
PrivacySweep screens text you provide using three layers:

1. **Regex** finds Indian 10-digit mobile-number patterns (starting with 6 to 9).
2. **Keyword detection** checks predefined public-exposure phrases.
3. **Machine learning** (TF-IDF + Logistic Regression) classifies the text as Review or No Review.

**Review categories:** 🔴 High Review Priority · 🟡 Needs Verification · 🟢 No Review

**Limitations**
- It analyzes only the text you enter. It does not search the internet or crawl directories.
- It does not verify that a number is valid, public, or who owns it.
- The model uses a small synthetic training dataset and has not been validated on real-world data.
- Probabilities are classifier scores, not calibrated real-world risk.
- Removal requests are drafts only. You must verify the listing and submit them yourself.

Built by **Swetha Boga** · [GitHub](https://github.com/Swetha-Boga/Privacy-Sweep-)
"""
    )

st.caption(
    "PrivacySweep · Privacy-awareness prototype · Results are preliminary and require "
    "human verification."
)
