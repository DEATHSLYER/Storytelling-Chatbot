from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
from rasa_sdk.events import SlotSet

class ActionGenerateStory(Action):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model.eval()  # Set model to evaluation mode

    def name(self):
        return "action_generate_story"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        input_text = tracker.latest_message.get("text", "")
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=500,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id
        )
        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        dispatcher.utter_message(text=story)
        return [SlotSet("generated_story", story)]


class ActionRefineStory(Action):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.model.eval()

    def name(self):
        return "action_refine_story"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        user_story = tracker.get_slot("generated_story") or "There is no existing story to refine."
        refined_prompt = f"Refine the following story to be more engaging and dramatic:\n{user_story}"
        inputs = self.tokenizer(refined_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=500,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id
        )
        refined_story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        dispatcher.utter_message(text=refined_story)
        return [SlotSet("generated_story", refined_story)]


class ActionStopStory(Action):
    def name(self):
        return "action_stop_story"

    async def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict):
        dispatcher.utter_message(text="The story session has been stopped. Let me know if there's anything else I can help you with!")
        return []
