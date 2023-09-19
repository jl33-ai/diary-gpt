import openai
# INSERT YOUR OPENAI API KEY HERE
openai.api_key = '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'

def therapist_conversation(transcript):

    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # or "gpt-4"
            messages=[{
                    "role": "system", 
                    "content": "You are a therapist talking to a patient with memory hoarding OCD. Continue the following conversation. Make sure you are ALWAYS asking a question for the patient to answer. Feel free to be colloquial and give your own thoughts."
                }, {
                    "role": "user", 
                    "content": '\n'.join(transcript)
                }],
            temperature=1   
        )
    response = completion.choices[0].message.content
    return response 

transcript = list()
memory = "This dinner is so nice, get to code, iMpossible. Thank You. So Much to tell her. Too good to be true, maybe."
transcript.append(f"Therapist: Let's take a look at some of your recent memories. You wrote '{memory}'")
transcript.append(f"Therapist: ")
while True: 
    response = therapist_conversation(transcript)
    print(response) 
    transcript[-1] = transcript[-1] + response
    print(transcript[0])
    print(transcript[1])    
    # Reprint transcript here
    user_input = str(input("Patient: "))
    if user_input.upper() == 'QUIT':
        break
    transcript.append(f"Patient: {user_input}")
    transcript.append(f"Therapist: ")

print("Full Transcript:")
for line in transcript:
    print(line)

# Save this convo 