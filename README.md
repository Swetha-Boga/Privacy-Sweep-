#  PrivacySweep
### Stop the Leak Before It Spreads.

**A Smart Personal Information Exposure Screening Tool Using Machine Learning**

Imagine posting an advertisement online with your phone number in it.

You might not even realize that your contact details are visible in the description.

What if a tool could read that description, spot possible personal information, and tell you which entries deserve a closer look?

That's the idea behind **PrivacySweep**.

PrivacySweep is a machine-learning-based privacy-awareness prototype that checks text for Indian mobile-number patterns and public-exposure keywords. It combines simple detection rules with machine learning to help users identify descriptions that may need further review.

---

##  01. The Real-World Problem

Think about these situations:

- Someone shares a business advertisement containing a mobile number.
- A listing description mentions that contact details are available in a public directory.
- A personal contact number appears in a profile description.

Finding such information manually can take time.

PrivacySweep helps by checking the text and highlighting possible warning signs.

**In simple words:** It works like a digital checking assistant that reads descriptions and points out things worth investigating.

---

##  02. What Exactly Does PrivacySweep Do?

You give PrivacySweep a piece of text.

It checks three things:

1. **Is there a mobile-number pattern?**
2. **Does the text contain words suggesting public exposure?**
3. **What does the machine-learning model predict about the text?**

It then displays a review category on the dashboard.

### A simple example

Imagine you enter:

> Our business contact details are available in a public directory. Call 9876543210.

PrivacySweep identifies:

- A possible Indian mobile-number pattern.
- The phrase "public directory".
- A reason to prioritize the description for review.

<badge color="warning">High Review Priority</badge>

This is an example of a detection result, not proof that the number is actually available online.

---

##  03. How Does It Think? Three Detection Layers

Think of PrivacySweep as having three checking assistants.

| Assistant | What it does | Simple explanation |
|---|---|---|
| Regex Detection | Finds mobile-number patterns | Looks for numbers matching a particular format |
| Keyword Detection | Finds exposure-related words | Looks for clues such as "public directory" |
| Machine Learning | Classifies descriptions | Learns patterns from example descriptions |

### Why do we need all three?

Imagine asking one person to check an entire document.

They might miss something.

Now imagine three people checking different things:

- One checks phone-number formats.
- One searches for exposure-related words.
- One studies the overall text pattern.

Together, they provide different signals that help the application decide what deserves attention.

That's the idea behind PrivacySweep's layered detection approach.

---

##  04. How Does PrivacySweep Work?

Here's the journey of one description through the application:

```text
       USER ENTERS TEXT
               |
               v
       MOBILE NUMBER CHECK
               |
               v
        KEYWORD CHECK
               |
               v
       MACHINE LEARNING
               |
               v
       REVIEW CATEGORY
               |
               v
       DASHBOARD RESULTS
               |
               v
      FURTHER HUMAN REVIEW
```

### Step-by-step explanation

**Step 1 – Input**

The user enters a listing description.

**Step 2 – Number Detection**

Regex checks whether the description contains a number matching the Indian mobile-number pattern used by the project.

**Step 3 – Keyword Detection**

The application checks for predefined phrases such as:
- Public directory
- Public listing
- Listed online
- Exposed online
- Public profile

**Step 4 – Machine Learning**

The text is converted into numerical information using TF-IDF. Logistic Regression then predicts whether the description belongs to the Review or No Review class.

**Step 5 – Final Categorization**

The application combines its detection results according to its screening rules and assigns a review category.

**Step 6 – Dashboard**

The user sees the result, summary information, and analysis history.

---

##  05. What Do the Three Results Mean?

PrivacySweep provides three review categories.

| Result | What it means | What should happen next? |
|---|---|---|
| 🔴 High Review Priority | A mobile-number pattern and exposure indicator were detected | Review the description carefully |
| 🟡 Needs Verification | A mobile-number pattern was detected without a positive exposure indicator | Check the context |
| 🟢 No Review | No mobile-number pattern was detected and the ML model predicted No Review | No review signal was identified by the current checks |

**Important:** These are screening categories, not confirmed privacy-risk levels. A result does not prove that personal information is publicly accessible.

---

## 🧠 06. Where Does Machine Learning Come Into the Picture?

Let's understand it with a simple example.

Imagine teaching a student to identify spam messages.

You show the student many examples:

- Some messages are marked as Review.
- Others are marked as No Review.

After studying those examples, the student starts learning patterns.

A machine-learning model works in a similar way.

PrivacySweep uses two important techniques:

### TF-IDF – Understanding Important Words

Suppose a description contains:

> Contact details are available in a public directory.

TF-IDF converts words into numerical values so the computer can work with the text.

In simple terms, it helps the computer represent which words are important in a description.

### Logistic Regression – Making a Prediction

After the text is converted into numbers, Logistic Regression uses patterns learned from training examples to predict one of two classes:

- Review
- No Review

The model does not understand privacy like a human. It learns statistical patterns from its training data.

### Why use Machine Learning?

Because descriptions are written in different ways. Machine learning gives the application another way to analyze text beyond searching for specific words.

---

##  07. What Can You Do With the Dashboard?

PrivacySweep is not just a Python script that prints a result.

It includes an interactive dashboard.

Its features include:

| Feature | Purpose |
|---|---|
| Text Analysis | Analyze an individual listing description |
| Review Categories | Display the screening result |
| Summary Statistics | Show analysis counts |
| Analysis History | View previous analysis records |
| CSV Batch Analysis | Analyze multiple descriptions together |
| Removal Request Assistant | Generate an editable removal-request draft |

### Why is CSV analysis useful?

Imagine a user has 100 listing descriptions.

Checking them one by one would take time.

With CSV batch analysis, multiple descriptions can be submitted together and processed in one workflow.

---

## ✉️ 08. What Happens After a High-Priority Result?

PrivacySweep also includes a removal-request assistant.

Imagine a user finds a listing that deserves further investigation.

They may want to contact the platform hosting that listing.

PrivacySweep can help generate an editable request draft.

The user can provide:
- Platform name
- Listing URL

The application prepares a draft that the user can review and edit.

**PrivacySweep does not automatically send the request or remove the listing.**

The user remains in control of the next step.

---

##  09. Why Is This Project Useful?

PrivacySweep explores how technology can support everyday privacy awareness.

It demonstrates how:

- Python can process text.
- Regex can identify specific patterns.
- NLP can convert words into numerical features.
- Machine learning can classify descriptions.
- Interactive dashboards can present results clearly.
- Automation can reduce repetitive checking work.

The objective is to make potential exposure indicators easier to notice and investigate.

---

## 🛠️ 10. Technology Used

| Technology | Why it is used |
|---|---|
| Python | Main programming language |
| Pandas | Handles structured data |
| NumPy | Supports numerical operations |
| Scikit-learn | Builds the machine-learning pipeline |
| TF-IDF | Converts text into numerical features |
| Logistic Regression | Performs text classification |
| Regular Expressions | Detects mobile-number patterns |
| IPyWidgets | Builds interactive dashboard controls |
| Joblib | Saves and loads the trained model |
| Jupyter Notebook | Development environment |

---

##  11. How to Run the Project

### Step 1 – Download the project

```bash
git clone https://github.com/Swetha-Boga/Privacy-Sweep-.git
```

### Step 2 – Open the project folder

```bash
cd Privacy-Sweep-
```

### Step 3 – Install the required libraries

If you use the provided requirements file:

```bash
pip install -r requirements.txt
```

### Step 4 – Open the notebook

Open the notebook file:

`notebooks_PrivacySweep_Clean_GitHub.ipynb`

### Step 5 – Run the notebook

Execute the notebook cells in order and interact with the dashboard.

**Environment note:** PrivacySweep was developed in a browser-based Jupyter/Pyodide environment. The standard Python installation instructions may require adjustments in a local Jupyter environment.

---

##  12. Project Structure

```text
Privacy-Sweep-/
│
├── notebooks_PrivacySweep_Clean_GitHub.ipynb
├── privacy_model.joblib
├── requirements.txt
└── README.md
```

Make sure these names match the actual files in the GitHub repository.

---

##  13. Project Limitations

PrivacySweep is a prototype, and it is important to understand what it cannot do.

- It analyzes only text provided by the user.
- It does not search the internet or crawl public directories.
- It does not verify whether a phone number is actually publicly exposed.
- It cannot identify the owner or validate the phone number.
- It uses a small synthetic training dataset.
- Its machine-learning model has not been validated on real-world data.
- Its classifications are not calibrated real-world risk probabilities.
- It does not automatically send removal requests or delete listings.

PrivacySweep should be treated as an initial screening assistant, not a complete privacy-protection service.

---

##  14. What's Next? – Future Improvements

PrivacySweep can grow further.

Some possible improvements include:

- Training and evaluating the model using a larger, more representative dataset.
- Adding detection for other types of personal information.
- Improving the explanation behind each classification.
- Adding more file formats for batch analysis.
- Improving dashboard usability.
- Developing a standalone, publicly accessible dashboard.

---

##  15. The Bigger Idea

PrivacySweep started with a simple question:

**Can we use machine learning to help people notice personal-information exposure indicators before they overlook them?**

This project explores one possible answer by combining pattern detection, text analysis, machine learning, and an interactive dashboard.

It is a practical demonstration of how multiple Python and machine-learning concepts can work together to solve a real-world-inspired problem.



**Swetha Boga**

[GitHub Profile](https://github.com/Swetha-Boga)

*PrivacySweep – A Machine Learning and Privacy-Awareness Prototype.*
