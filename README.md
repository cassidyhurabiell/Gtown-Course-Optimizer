# Gtown-Course-Optimizer
A personalized course scheduler and degree-planning system that combines transcript parsing, professor analysis, and student preferences to generate optimized, human-centered schedules. *individual project, not affiliated with Georgetown

This project began with a simple frustration: most university scheduling tools treat students like spreadsheets. They can tell you whether classes overlap, but they cannot tell you whether you will actually enjoy a professor, burn out from your schedule, stay on track for graduation, or discover classes that genuinely excite you. I wanted to build something that approached scheduling more like an advisor would, balancing graduation requirements with personality, interests, workload, and long-term goals.

The optimizer generates schedules tailored to each individual student. Instead of only maximizing “efficiency,” it tries to create schedules that students will realistically enjoy and succeed in. A student interested in fintech who prefers compact morning schedules should not receive the same recommendations as someone exploring philosophy and prioritizing discussion-heavy seminars with highly rated professors.

The system combines several different layers of information. It can parse a student transcript PDF to determine completed coursework, analyze Georgetown course data, interpret natural-language scheduling preferences, and integrate professor information scraped from RateMyProfessor. From there, it generates multiple schedule options with different priorities, such as balanced schedules, requirement-heavy schedules, or more exploratory schedules focused on interests and professor quality.

One of the more important aspects of the project is that it was designed specifically around Georgetown’s scheduling structure. Georgetown uses modular half-semester courses alongside traditional semester-long classes, which creates edge cases that many generic schedulers ignore. Rather than simply checking whether two classes occur at the same time, this project also considers whether their actual date ranges overlap. The optimizer also handles linked sections, varying credit structures, and grouped degree requirements such as elective “buckets,” where students may need a certain number of courses from a category rather than one specific class.

The recommendation system is intended to feel more human than algorithmic. Students can describe what they want in plain English using inputs like:

> “Avoid late classes, prioritize compact schedules, and recommend courses related to entrepreneurship and AI.”

The system then converts those preferences into weighted scheduling priorities. It also attempts to explain recommendations rather than simply outputting a schedule. Professor reviews, tags, and ratings are used not only for ranking courses, but also for generating reasoning behind recommendations, such as highlighting engaging teaching styles, practical applications, or discussion-based classrooms that align with a student’s preferences.

The project is built primarily with Python, pandas, and Streamlit. Much of the work went into data cleaning and modeling rather than UI design. Georgetown schedule exports contain messy meeting information, partial-term date ranges, linked sections, and inconsistent formatting, so a significant portion of the project involved building parsers capable of converting raw course data into structured scheduling logic. RateMyProfessor data is scraped and cached locally to reduce repeated requests and improve performance.

Because much of the underlying data comes from institutional exports and personal academic information, this repository does not include Georgetown course CSVs, student transcripts, or requirement datasets. Users will need to provide their own local data files. The repository is structured so that these datasets can be plugged into the optimizer without exposing private information publicly on GitHub.

At the moment, the project focuses primarily on single-semester optimization, though future directions include multi-semester planning, study abroad-aware scheduling, workload prediction, internship integration, social scheduling features, and more advanced recommendation systems. I am also interested in improving the natural-language preference modeling so that the system can better understand how students describe their ideal college experience.

The project can be run locally through Streamlit:

```bash id="n7v3xw"
pip install -r requirements.txt
streamlit run app.py
```

This repository is still actively evolving, and there are definitely rough edges. Some Georgetown-specific assumptions are hardcoded, the preference parser is still improving, and modular overlap handling occasionally produces strange edge cases. Still, the project has become a much larger exploration of personalized academic planning than I originally expected.

Ultimately, the goal is not just to generate schedules. It is to help students build semesters that fit who they are and who they want to become.

---

### Disclaimer

This project is independent and unofficial. It is not affiliated with or endorsed by Georgetown University.
