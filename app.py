import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Your paragraph goes here - REPLACE THIS WITH YOUR OWN PARAGRAPH
YOUR_PARAGRAPH = """
Esperanza conducts pre-employment background checks on all applicants who accept an offer of employment. All offers of employment or volunteering at Esperanza are contingent upon clear results of a thorough background check. Background checks will be conducted for or requested of individuals in the following circumstances:  Job candidates to whom offers of employment have been made  Employees who are being promoted into new positions, as deemed necessary by the individual requirements of the position  Current employees whose current criminal background checks have aged will be checked every three years. Employees will be notified by HR of their expiring clearances and given instructions and a deadline for completing them.  Child abuse and Criminal: Mental Health professionals, Clinicians, nurses, social workers, medical assistants, dietitians, and any other employee whose job involves regular and repeated contact with children.  Criminal only: All other employees  For job applicants, background checks will not be requested or conducted during the employment application process, but only after a conditional offer of employment has been made in compliance with Pennsylvania law.

Pre-employment background checks will include:  Pennsylvania (PA) Criminal Background clearance  Child Abuse History check (when applicable to the position): PA Child Abuse clearance, and FBI Fingerprint-based clearance  Social Security Verification: validates the applicant's Social Security number, date of birth and former addresses.  Personal and Professional References: calls will be placed to individuals listed as references by the applicant.  Medicare/Medicaid Exclusion Checks: Checks for Medicare/Medicaid fraud history. 

An HR representative will contact employees regarding the renewal of aged clearances. Background check information will be kept on file by HR for at least five years. Final candidates for employment and volunteers will complete the background check authorization form and return it to Human Resources (HR). Human Resources will order the background checks upon receipt of the signed release form, and either internal HR staff or an employment screening service will conduct the checks. A designated HR representative will review all results.

If there are no negative results, the applicant or volunteer will continue with the onboarding process. In instances where negative or incomplete information is obtained, the hiring manager and the Director of Human Resources will conduct an assessment to determine the potential risks and liabilities related to the job's requirements and make a recommendation to the CEO regarding hiring or not hiring the individual. If a decision is made by the CEO to not hire or promote a candidate based on the result of the background check, the decision should be documented in writing and signed by the CEO. HR will follow all applicable laws regarding notifying the candidate of the rejection. If information obtained in a background check leads Esperanza to deny employment, a copy of the report will be provided to the applicant, and the applicant will have the opportunity to dispute the report’s accuracy. All background checks are conducted in conformity with the Federal Fair Credit Reporting Act, the Americans with Disabilities Act, and state and federal privacy and anti-discrimination laws. Reports are kept confidential and are only viewed by individuals involved in the hiring process. Licensed employees or contractors will be subject to other background checks as required by law and/or government contract recommendations. Sources for these reviews may include, but not limited to, System for Award Management (SAM), Office of Inspector General List of Excluded Individuals/Entities, National Practitioner Data Bank, and Pennsylvania Medicheck.

Esperanza has developed employment classifications for employees to understand their employment status. These classifications are as follows: 1. Full-time employees are those whose regularly scheduled work week consists of 32 or more hours. Employees who work 32 hours a week or more are eligible to participate in all employee benefits described more fully in this Handbook and/or the Human Resources Policy and Procedure Manual. 2. Part-time employees are those whose regularly scheduled work week consists of fewer than 32 hours per week. Part-time permanent employees are eligible to participate in employee health benefits if they are scheduled to work 24 hours a week or more. Employees scheduled to work less than 24 hours per week do not qualify to participate in group health, dental, life, and long-term disability insurance. However, all part-time employees can participate in the Esperanza Health Center 401(k) plan and, in certain cases more fully described in the Esperanza Health Center 401(k) Summary Plan Description, are eligible to receive Esperanza matching or discretionary contributions. Part-time employees that work extra hours across departments on an occasional, voluntary basis will only receive benefits entitled to them as per their hired position.

3. Per-diem employees are those employees who work less than 24 hours a week or are employed to fill in for other special needs. Such employees are not eligible to participate in employee benefits. 4. Temporary employees are employees hired to work for less than 12 months, either full or part- time. Part-time temporary employees are not eligible to participate in employee benefits. Temporary employees who work more than 30 hours per week are eligible for health insurance after 90 days (about 3 months) if upon hire, the position is designated to be 30 or more hours per week. If the number of hours per week is not known, the temporary employee will be considered a variable hour employee and subject to the rules of that employment classification. Temporary employees are entitled to sick leave according to Philadelphia law and proportionate to the hours they are scheduled to work. Sick paid time off is accrued from the date of hire to a maximum of seven days per year. 5. Probationary employees are those who have not yet completed 90 days (about 3 months) of employment with Esperanza. They accrue any vacation time, personal holiday, and sick time benefits for which they are eligible during this time but may not use these benefits until after the satisfactory completion of their probationary period. Probationary employees may be paid at a lower rate during their training period. 6. Conditional employees are those placed on conditional status as a disciplinary measure as discussed in Policy 9.2, Progressive Discipline. 7. Variable hour employees are those who, upon their start date, it cannot be determined that the employee is expected to work on average at least 8. 30 hours per week. A look- back measure of 6 months (following the 1000-hour rule) will be used to assess whether the variable hour employee will receive health insurance benefits. 

In addition, all positions are classified as exempt or non-exempt status based upon the Federal and state wage and hour laws: 1. Non-exempt hourly employees are those in support staff positions as defined by the Fair Labor Standards Act (FLSA). These employees are paid hourly and entitled to overtime pay at time and a half, if asked to work over 40 hours in the work week by Esperanza. All overtime must be approved by the employee’s supervisor prior to working. Non-exempt employees receive a one-hour lunch break; 30 minutes are unpaid and the remaining 30 minutes are designated as two 15- minute breaks. 2. Non-exempt salaried employees are those in administrative and managerial positions as defined by the FLSA. These employees are paid a salary and entitled to overtime pay at time and a half, if requested to work over 40 hours in the work week by Esperanza. Non- exempt salaried employees receive a half-hour lunch break, which is unpaid. 3. Exempt employees are those in executive, administrative, professional, and other positions as defined by the FLSA. These employees are paid a salary and are exempt from overtime. Exempt employees receive a half-hour lunch break, which is unpaid.

Esperanza will use volunteers sometimes to fulfill the health center's needs and provide educational and training opportunities to students, interns, and residents. This policy outlines the documentation and orientation requirements for volunteer positions and student rotations before starting service.  All volunteers must complete the Esperanza Health Center Volunteer or Student Rotation Application; complete an orientation of Esperanza policies; and complete and sign a HIPAA Confidentiality statement. Depending upon the service provided, volunteers may be requested to provide a copy of their Pennsylvania Criminal Background clearance.  Clinical volunteers for patient care and support positions (including students and residents) must provide:  Copies of valid, current licenses, certifications (if applicable) and CPR certificate  Evidence of OSHA training (or complete Esperanza’s OSHA training and post-test)  Evidence of current immunizations (HepB and TB)  Evidence of current liability insurance All documentation should be provided before the start of the assignment. For all volunteers and students covered by this policy, a file must be kept by the manager or supervisor and the application should be kept on file with the Human Resources Manager’s records in the Human Resources department. Volunteers, interns, and student residents may be issued an Esperanza identification (ID) card. When an ID card is required, students and/or volunteers must report to Human Resources to obtain the ID card and for the completion of onboarding.

If a former Esperanza employee returns to Esperanza after a break in service, his or her new start date will be the date of rehire. Rehired employees are probationary employees for the first 90 days of their employment, but are not eligible to use their accrued vacation, personal day, or sick paid time off until the probationary period is completed. Returning employees may or may not be rehired for the same position or same pay.

Esperanza will provide assistance to new employees in their transition into their new position at Esperanza. The Human Resources Department conducts new Employee Orientation and Onboarding. During onboarding, the employee must provide proof of identification for employment verification and complete all administrative paperwork including the following: job description review, facility tour, W-4 Employment Withholding Form, I-9 Employment Verification, and the Authorization for Direct Deposit form. OSHA Training is completed during the first 10 days of employment. Employees must complete the OSHA post-test to ensure there is a proper understanding of EHC safety policies and procedures. Orientation will normally be conducted within the employee’s first month of employment. Esperanza orientation includes an overview of Esperanza’s personnel policies and procedures, HIPAA training, information technology and electronic medical records training, health benefits orientation, and 401(k) plan orientation. Additional training and orientation will be conducted by the employee’s department for job specific training. Certain positions require participation in continuing education and training programs when such instruction is considered necessary for satisfactory job performance, licensure, or credentials. Time spent in orientation and onboarding is considered working hours and employees will be compensated for that time. Employees are encouraged to discuss questions about orientation and onboarding with their supervisor at any time. 

Esperanza expects employees to wear an Esperanza identification (ID) card to promote safety and uniformity. The identification card should be visible so all can view and allow staff to be identified as an Esperanza employee. Obstructions of your ID card should be avoided. An ID card will be issued on the first day of employment during the Human Resources Orientation. Employees are expected to wear their Esperanza ID card at all times when working. The ID card should be free from distracting stickers, pens, or anything that would conflict with the image of Esperanza’s mission. Failure to adhere to this can lead to disciplinary action. Replacement of lost identification cards will cost the employee $5.00. The ID card is the property of Esperanza and must be collected by the manager or Human Resources Director during the exit interview or when an employee is terminated. All ID cards will be returned to the Human Resources Department after termination of employment.

Esperanza typically adheres to a 40-hour work week for employees. This policy outlines the hours of operation during which the health center provides services and the regular work schedule for non- exempt and exempt employees. Esperanza’s clinical hours are Monday through Saturday. A work week begins Sunday at 12:00 a.m. and ends Saturday at 11:59 p.m. Some weekend and evening hours may be required from all employees. Hourly employees cannot begin work or perform any work related duties prior to the beginning of the normal work day (8:30 a.m. to 5:00 p.m.), unless specific authorization from the supervisor is given. Similarly, hourly employees cannot continue work beyond the end of their normal day without specific authorization from their supervisor. Exempt employees are hired to complete specific job duties and tasks that are associated with a job position without regard to specific hours. Exempt employees are expected to work the hours necessary to complete their work without the expectation of additional compensation. All employees must understand that, because of unusually heavy patient volume, special projects or emergencies, extra work time may occasionally be required. In these instances, which may occur with or without advance notice, staff members will be expected to fulfill their responsibilities as necessary to ensure good patient care. 
Esperanza desires to foster a work environment of excellence in which timeliness and consistent attendance is the professional expectation of all employees. Esperanza encourages all employees to have habits of good attendance and punctuality. Employees are expected to report to and be prepared to work by 8:30 a.m. and attend devotions, prayer groups, or other scheduled meetings between 8:30 and 9:00 a.m. unless previous authorization is given from the employee’s supervisor. Employees are not allowed to eat breakfast on work time unless approval from a supervisor is granted. Unauthorized absence(s) or tardiness will not be tolerated and may result in disciplinary action or dismissal. 

Procedure – Unscheduled Absence: At times, circumstances beyond an employee’s control may result in an unexpected absence from work for either all or part of a day. In these cases, the procedure outlined in this paragraph should be followed and supporting documentation may be requested by the employee’s supervisor or the Human Resources Director. To maintain a work environment of excellence, employees are expected to clearly and proactively communicate with their supervisor if they will be absent. If an employee is unable to report for work on a given day, he or she is required to call and speak personally with his or her supervisor no later than one hour before the scheduled starting time, unless alternative arrangements have been arranged with the employee’s supervisor. When speaking with their supervisor, employees should describe the reason for the absence. Because of the need to plan for covering open positions, messages via the supervisor’s voice mail or other third parties are not acceptable. Text communication for call-outs is allowed as determined by each department manager. Communication should occur as soon as possible but no later than 7:30 a.m. so that alternate staffing arrangements can be made. Text, as well as oral communication, should 1) be two-way; 2) close the loop about duties during the day that need to be completed in the employee’s absence; 3) be prompt; and 4) be oral, if needed, to close the loop. The employee should keep his/her phone on and be accessible to talk with the supervisor if needed. 

If employees are unable to reach their supervisor, they should contact and communicate with the next person who is designated by the supervisor. Failure to communicate directly to someone may result in a disciplinary action. If after several attempts, the employee is not able to communicate directly to a manager, he or she may leave a voicemail for the supervisor, but must follow up with a phone call at the beginning of the business day to personally inform his or her manager of the absence.

Unpaid time off is approved only under extraordinary circumstances (see 6.2 Personal and Medical Leave) and with available coverage for clinic functions. Requests for unpaid time off will be reviewed by the employee’s supervisor and the Human Resources Director for possible approval only after all accrued sick, vacation, personal, or other paid time off has been expended. Supporting documentation may be requested by the employee’s supervisor or the Human Resources Director Employees who are absent from work for three consecutive days without notice or communication with their supervisor or human resources will be considered as having quit voluntarily.

Disciplinary action will be taken in the event that an employee has four unscheduled absences, within a 60-day time period, unless prior approval for reporting late to work for activities such as attending doctor’s appointments, court hearings, school meetings, or other personal requests was given by the department supervisor 24 hours in advance. The progressive disciplinary actions for more than four unscheduled absences in a 60-day time period will include the following: 1. First verbal warning. 2. Second verbal warning. 3. Written warning. 4. First loss of vacation time accrued within the current pay period. 5. Second loss of vacation time accrued within the current pay period. 6. Review by CEO, Human Resources Director, and immediate supervisor, with employees placed in a 60-day conditional period. As stated in Section 9.2 Progressive Discipline, during the conditional period, the employee will receive full salary and benefits. If the problem is not corrected during the conditional period, employment may be terminated. If, after a successful conditional period the problem reoccurs, the employee may be terminated. 7. Temporary suspension. As stated in Section 9.2 Progressive Discipline, an employee may be temporarily suspended or permanently dismissed for the following reasons, including but not limited to, poor job performance or failure to comply with the policies and procedures of Esperanza. This is a time without pay and all accrual of benefit time will also be suspended. 8. Decision by CEO, Human Resources Director, and immediate supervisor whether to continue to employ or dismiss the employee. As stated in Section 9.2 Progressive Discipline, an employee may be dismissed at any time, including just cause as discussed in Rules of Conduct. Dismissal will be effective immediately. 


"""

# Set up the page
st.set_page_config(page_title="Simple RAG Chatbot", page_icon="🤖")
st.title("🤖 Simple RAG Chatbot")
st.write("Ask me questions about the paragraph below!")

# Show the paragraph
with st.expander("📖 Click to see the source paragraph"):
    st.write(YOUR_PARAGRAPH)

# Initialize the embedding model (this runs once)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Split paragraph into sentences and create embeddings
@st.cache_data
def prepare_data(paragraph):
    # Split into sentences
    sentences = re.split(r'[.!?]+', paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Create embeddings
    model = load_model()
    embeddings = model.encode(sentences)
    
    return sentences, embeddings

# Get OpenAI API key from user
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Enter your OpenAI API Key:", 
        type="password", 
        value=st.session_state.openai_api_key,
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    if api_key:
        st.session_state.openai_api_key = api_key
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Prepare the data
sentences, embeddings = prepare_data(YOUR_PARAGRAPH)

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the paragraph..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check if API key is provided
    if not st.session_state.openai_client:
        with st.chat_message("assistant"):
            st.error("Please enter your OpenAI API key in the sidebar to use the chatbot.")
    else:
        # Find relevant sentences
        model = load_model()
        question_embedding = model.encode([prompt])
        
        # Calculate similarity
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        # Get top 2 most similar sentences
        top_indices = np.argsort(similarities)[-2:][::-1]
        relevant_sentences = [sentences[i] for i in top_indices if similarities[i] > 0.1]
        
        if not relevant_sentences:
            relevant_sentences = [sentences[0]]  # Fallback to first sentence
        
        # Create context
        context = " ".join(relevant_sentences)
        
        # Generate response using OpenAI
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You are a helpful assistant. Answer the user's question based on this context: {context}. If the context doesn't contain enough information to answer the question, say so politely."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    answer = response.choices[0].message.content
                    st.write(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Show which sentences were used (for debugging)
                    with st.expander("🔍 Source sentences used"):
                        for sentence in relevant_sentences:
                            st.write(f"• {sentence}")
        
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error: {str(e)}")
                st.write("Make sure your OpenAI API key is correct and you have credits available.")

# Instructions
with st.sidebar:
    st.header("📝 Instructions")
    st.write("""
    1. Get an OpenAI API key from https://platform.openai.com/api-keys
    2. Enter your API key in the box above
    3. Ask questions about the paragraph
    4. The chatbot will find relevant sentences and answer your questions!
    """)
    
    st.header("💡 Tips")
    st.write("""
    - Ask specific questions about the content
    - Try questions like "What is the Amazon rainforest?" or "Why is deforestation a problem?"
    - The chatbot can only answer based on the paragraph provided
    """)
