import os

# Suppress TensorFlow warnings for cleaner output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pipeline import run_resume_screening_pipeline


def main() -> None:
    # ==================== CONFIGURATION ====================
    # Path to resume PDF (change this to test different resumes)
    resume_path = "BansariNaik-thirdyr.pdf"
    
    # Job description to match against
    job_description = """
    Your Role

Design, develop, test, and deploy Java-based applications.
Write clean, maintainable, and efficient code following best practices.
Angular 8+ / React (including RxJS)
JavaScript, jQuery
Bootstrap, Material Design, CSS, HTML5 
Participate in the full software development lifecycle (SDLC).
Develop RESTful APIs and integrate with external systems.
Collaborate with product owners, architects, QA teams, and fellow developers.
Troubleshoot, debug, and resolve application-related issues.
Optimize application performance and scalability.
Maintain technical documentation and participate in code reviews.
Work with DevOps teams to automate deployment and CI/CD processes.
Ensure security and compliance across the application lifecycle.

Your profile

Strong proficiency in Java (8/11/17) and object‑oriented programming.
Experience with Spring Framework / Spring Boot.
Understanding of REST APIs, microservices architecture.
Strong knowledge of JPA/Hibernate or similar ORM frameworks.
Angular 8+ / React (including RxJS)
JavaScript, jQuery
Bootstrap, Material Design, CSS, HTML5
IDEs: Visual Studio Code, Eclipse
Hands‑on experience with databases: MySQL / PostgreSQL / Oracle / MongoDB.
Proficiency in Git and version control workflows. Experience with Maven/Gradle build tools.
Familiarity with JUnit, Mockito, or other testing frameworks.
Knowledge of Agile/Scrum methodologies.
    """
    
    try:
        # Header
        print("\n" + "=" * 75)
        print("RESUME MATCHING - LSTM & CNN DEEP LEARNING MODELS")
        print("=" * 75)
        print(f"\nResume File: {resume_path}")
        print(f"Job Role: Full-Stack Java Developer\n")
        
        # Run the screening pipeline
        result = run_resume_screening_pipeline(resume_path, job_description)
        
        # Display Results
        print("=" * 75)
        print("DEEP LEARNING SCORING")
        print("=" * 75)
        
        lstm_score = result['lstm_score']
        cnn_score = result['cnn_score']
        final_score = result['final_match_score']
        
        # Visual representation of scores (LSTM and CNN only - NO TF-IDF)
        print(f"\n[LSTM] Sequential Understanding Score:  {lstm_score:>6}%")
        print(f"[CNN]  Pattern Recognition Score:      {cnn_score:>6}%")
        print("-" * 75)
        print(f"[FINAL] Match Score (Average):         {final_score:>6}%")
        
        # Score interpretation
        print("\n" + "=" * 75)
        print("SCORE INTERPRETATION")
        print("=" * 75)
        
        if final_score >= 70:
            status = "[EXCELLENT MATCH]"
            description = "Resume strongly aligns with job requirements"
        elif final_score >= 50:
            status = "[GOOD MATCH]"
            description = "Resume has relevant skills but some gaps exist"
        elif final_score >= 30:
            status = "[MODERATE MATCH]"
            description = "Resume has some relevant experience but significant gaps"
        else:
            status = "[POOR MATCH]"
            description = "Resume lacks critical skills for this role"
        
        print(f"\nStatus: {status}")
        print(f"Assessment: {description}")
        
        # Skills Analysis
        print("\n" + "=" * 75)
        print("SKILLS ANALYSIS")
        print("=" * 75)
        
        matched_skills = result['matched_skills']
        missing_skills = result['missing_skills']
        
        print(f"\n[MATCHED] Skills found in resume ({len(matched_skills)}):")
        if matched_skills:
            for skill in matched_skills:
                print(f"   [+] {skill.capitalize()}")
        else:
            print("   None detected")
        
        print(f"\n[MISSING] Skills NOT in resume ({len(missing_skills)}):")
        if missing_skills:
            for skill in missing_skills:
                print(f"   [-] {skill.capitalize()}")
        else:
            print("   All required skills are present!")
        
        # Summary
        print("\n" + "=" * 75)
        print("RECOMMENDATION")
        print("=" * 75)
        total_required = len(matched_skills) + len(missing_skills)
        match_percentage = (len(matched_skills) / total_required * 100) if total_required > 0 else 0
        
        print(f"\nSkill Coverage: {match_percentage:.1f}% ({len(matched_skills)}/{total_required})")
        print(f"Deep Learning Score: {final_score}%")
        print(f"  - LSTM Model: {lstm_score}%")
        print(f"  - CNN Model:  {cnn_score}%")
        
        if final_score >= 70 and match_percentage >= 70:
            print("\n[VERDICT] STRONG CANDIDATE - Proceed with interview")
        elif final_score >= 50 or match_percentage >= 50:
            print("\n[VERDICT] POTENTIAL CANDIDATE - May need upskilling in missing areas")
        else:
            print("\n[VERDICT] NOT RECOMMENDED - Significant skill gaps exist")
        
        print("\n" + "=" * 75 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
