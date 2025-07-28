#!/usr/bin/env python3
"""
General English Test Suite for Attention Module - Top K Facts Retrieval

This test suite evaluates the attention module using common English paragraphs 
and questions, similar to datasets used in ATTRIEVAL and reading comprehension research.
Tests cover various domains like news, history, science, technology, etc.
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Import attention_viz if available
try:
    from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig
    ATTENTION_VIZ_AVAILABLE = True
    print("✅ attention_viz module loaded successfully")
except ImportError as e:
    print(f"❌ attention_viz not available: {e}")
    ATTENTION_VIZ_AVAILABLE = False

@dataclass
class TestExample:
    """A test example with context, question, and expected relevant facts"""
    id: str
    context: str
    question: str
    expected_answer: str
    ground_truth_facts: List[str]  # Facts that should be identified as relevant
    irrelevant_facts: List[str]    # Facts that should NOT be identified as relevant
    description: str
    domain: str  # Domain category (news, history, science, etc.)

@dataclass
class AttentionTestResult:
    """Results from testing attention on a single example"""
    example_id: str
    retrieved_facts: List[Dict]
    ground_truth_facts: List[str]
    irrelevant_facts: List[str]
    precision: float
    recall: float
    f1_score: float
    top_k_accuracy: float
    attention_scores: Dict

class GeneralEnglishAttentionTestSuite:
    """Test suite for evaluating attention-based fact retrieval on general English text"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", k: int = 5):
        self.model_name = model_name
        self.k = k
        self.model = None
        self.tokenizer = None
        self.retriever = None
        self.test_examples = []
        self.results = []
        
    def setup_model(self):
        """Setup the model and attention analysis components"""
        if not ATTENTION_VIZ_AVAILABLE:
            raise RuntimeError("attention_viz not available")
            
        print(f"🔧 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with forced eager attention implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",  # Use CPU for consistent testing
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Force eager attention for attention extraction
        )
        self.model.eval()
        
        # Ensure attention outputs are enabled and using eager implementation
        self.model.config.output_attentions = True
        self.model.config._attn_implementation = "eager"
        
        # Setup attention analysis
        extractor = AttentionExtractor(self.model, self.tokenizer)
        config = AttrievelConfig(
            layer_fraction=0.25,
            top_k=self.k,
            frequency_threshold=0.95,
            max_facts=self.k * 2  # Allow more facts to be considered
        )
        self.retriever = AttrievelRetriever(extractor, config)
        
        print("✅ Model and attention components loaded successfully")
        
    def create_test_dataset(self):
        """Create a comprehensive test dataset with diverse English contexts"""
        self.test_examples = [
            # News/Current Events
            TestExample(
                id="tech_news_1",
                context=(
                    "Technology News Update:\n"
                    "Apple announced its latest iPhone 15 Pro at the annual September event yesterday. "
                    "The new device features a titanium frame, improved camera system with 48MP main sensor, "
                    "and the new A17 Pro chip built on 3nm technology. The phone will be available in four colors: "
                    "Natural Titanium, Blue Titanium, White Titanium, and Black Titanium. "
                    "Pricing starts at $999 for the 128GB model, with 256GB and 512GB options also available. "
                    "Pre-orders begin Friday, September 15th, with general availability starting September 22nd. "
                    "The company also announced updates to the Apple Watch Series 9 and AirPods Pro. "
                    "CEO Tim Cook emphasized the environmental benefits of the new titanium construction."
                ),
                question="What material is used for the iPhone 15 Pro frame?",
                expected_answer="titanium",
                ground_truth_facts=[
                    "The new device features a titanium frame",
                    "iPhone 15 Pro",
                    "titanium construction"
                ],
                irrelevant_facts=[
                    "48MP main sensor",
                    "A17 Pro chip built on 3nm technology",
                    "Pricing starts at $999",
                    "Pre-orders begin Friday, September 15th",
                    "Apple Watch Series 9",
                    "AirPods Pro",
                    "CEO Tim Cook"
                ],
                description="Technology news requiring identification of specific product features",
                domain="technology"
            ),
            
            # Historical Context
            TestExample(
                id="history_wwii_1",
                context=(
                    "World War II in the Pacific:\n"
                    "The attack on Pearl Harbor occurred on December 7, 1941, when Japanese forces "
                    "launched a surprise assault on the US naval base in Hawaii. Admiral Isoroku Yamamoto "
                    "planned the operation, which involved over 350 aircraft from six aircraft carriers. "
                    "The attack lasted approximately two hours and resulted in the destruction of "
                    "numerous battleships including the USS Arizona and USS Oklahoma. Over 2,400 Americans "
                    "were killed and 1,200 wounded. President Franklin D. Roosevelt called it 'a date which "
                    "will live in infamy' in his speech to Congress the following day. The attack prompted "
                    "the United States to formally enter World War II, joining the Allied forces against "
                    "the Axis powers including Germany, Italy, and Japan."
                ),
                question="Who planned the Pearl Harbor attack?",
                expected_answer="Admiral Isoroku Yamamoto",
                ground_truth_facts=[
                    "Admiral Isoroku Yamamoto planned the operation",
                    "Japanese forces launched a surprise assault",
                    "attack on Pearl Harbor"
                ],
                irrelevant_facts=[
                    "December 7, 1941",
                    "US naval base in Hawaii",
                    "over 350 aircraft from six aircraft carriers",
                    "USS Arizona and USS Oklahoma",
                    "Over 2,400 Americans were killed",
                    "President Franklin D. Roosevelt",
                    "a date which will live in infamy"
                ],
                description="Historical fact retrieval requiring identification of specific person responsible",
                domain="history"
            ),
            
            # Science/Nature
            TestExample(
                id="science_climate_1",
                context=(
                    "Climate Change and Ocean Currents:\n"
                    "The Gulf Stream is a powerful ocean current that originates in the Gulf of Mexico "
                    "and flows along the eastern coast of the United States before crossing the Atlantic "
                    "to Europe. This current transports warm water northward, significantly affecting "
                    "the climate of Western Europe, making it much warmer than other regions at similar "
                    "latitudes. Scientists have observed that rising global temperatures due to climate "
                    "change are causing the Greenland ice sheet to melt at an accelerated rate. The "
                    "resulting freshwater influx into the North Atlantic is disrupting the density-driven "
                    "circulation patterns that power the Gulf Stream. Dr. Sarah Matthews from the Woods Hole "
                    "Oceanographic Institution warns that a weakening Gulf Stream could lead to cooler "
                    "temperatures in Europe and rising sea levels along the US East Coast."
                ),
                question="What is causing disruption to the Gulf Stream circulation?",
                expected_answer="Greenland ice sheet melting and freshwater influx",
                ground_truth_facts=[
                    "Greenland ice sheet to melt at an accelerated rate",
                    "resulting freshwater influx into the North Atlantic",
                    "disrupting the density-driven circulation patterns",
                    "rising global temperatures due to climate change"
                ],
                irrelevant_facts=[
                    "originates in the Gulf of Mexico",
                    "flows along the eastern coast of the United States",
                    "making it much warmer than other regions",
                    "Dr. Sarah Matthews from the Woods Hole Oceanographic Institution",
                    "cooler temperatures in Europe"
                ],
                description="Science explanation requiring multi-step causal reasoning",
                domain="science"
            ),
            
            # Literature/Arts
            TestExample(
                id="literature_shakespeare_1",
                context=(
                    "Shakespeare's Literary Legacy:\n"
                    "William Shakespeare, often regarded as the greatest writer in the English language, "
                    "was born in Stratford-upon-Avon in 1564. During his career, he wrote approximately "
                    "37 plays and 154 sonnets. His works are traditionally divided into three periods: "
                    "the early period (1590-1595) featuring comedies like 'A Midsummer Night's Dream', "
                    "the middle period (1595-1605) containing his greatest tragedies including 'Hamlet', "
                    "'Othello', 'King Lear', and 'Macbeth', and the late period (1605-1613) which included "
                    "romances such as 'The Tempest'. The Globe Theatre in London was closely associated "
                    "with Shakespeare's company, the Lord Chamberlain's Men, later known as the King's Men. "
                    "Shakespeare died in 1616, but his influence on literature, theater, and the English "
                    "language continues to this day."
                ),
                question="Which period of Shakespeare's career included his greatest tragedies?",
                expected_answer="middle period (1595-1605)",
                ground_truth_facts=[
                    "middle period (1595-1605)",
                    "containing his greatest tragedies",
                    "'Hamlet', 'Othello', 'King Lear', and 'Macbeth'"
                ],
                irrelevant_facts=[
                    "born in Stratford-upon-Avon in 1564",
                    "37 plays and 154 sonnets",
                    "early period (1590-1595)",
                    "'A Midsummer Night's Dream'",
                    "late period (1605-1613)",
                    "'The Tempest'",
                    "Globe Theatre in London",
                    "died in 1616"
                ],
                description="Literature analysis requiring temporal categorization",
                domain="literature"
            ),
            
            # Economics/Business
            TestExample(
                id="economics_inflation_1",
                context=(
                    "Economic Analysis Report:\n"
                    "The Federal Reserve announced a 0.25% interest rate increase yesterday, bringing "
                    "the federal funds rate to 5.5%, the highest level in 22 years. This decision was "
                    "made to combat persistent inflation, which has remained above the Fed's 2% target "
                    "for the past 18 months. Consumer prices rose 3.8% year-over-year in the latest "
                    "report, driven primarily by increases in housing costs, energy prices, and food. "
                    "Fed Chair Jerome Powell stated that the central bank remains committed to bringing "
                    "inflation back to target levels, even if it means accepting some economic slowdown. "
                    "Stock markets reacted negatively to the news, with the S&P 500 falling 1.2% and "
                    "the Dow Jones dropping 0.8%. Economists predict that mortgage rates could reach "
                    "7.5% by the end of the year, potentially cooling the housing market further."
                ),
                question="What is driving the current inflation according to the report?",
                expected_answer="housing costs, energy prices, and food",
                ground_truth_facts=[
                    "driven primarily by increases in housing costs, energy prices, and food",
                    "Consumer prices rose 3.8% year-over-year",
                    "persistent inflation"
                ],
                irrelevant_facts=[
                    "0.25% interest rate increase",
                    "federal funds rate to 5.5%",
                    "highest level in 22 years",
                    "Fed's 2% target",
                    "Fed Chair Jerome Powell",
                    "Stock markets reacted negatively",
                    "S&P 500 falling 1.2%"
                ],
                description="Economic analysis requiring identification of causal factors",
                domain="economics"
            ),
            
            # Geography/Travel
            TestExample(
                id="geography_amazon_1",
                context=(
                    "The Amazon Rainforest Ecosystem:\n"
                    "The Amazon rainforest spans across nine countries in South America, with Brazil "
                    "containing approximately 60% of the forest. This vast ecosystem covers about "
                    "5.5 million square kilometers and is home to an estimated 10% of all known species "
                    "on Earth. The Amazon River, which flows through the forest, is the world's largest "
                    "river by volume and the second-longest after the Nile. The rainforest plays a "
                    "crucial role in global climate regulation, absorbing approximately 2.2 billion tons "
                    "of carbon dioxide annually. However, deforestation rates have been alarming, with "
                    "satellite data showing that 11,568 square kilometers were cleared in 2022 alone. "
                    "The main drivers of deforestation include cattle ranching, soy cultivation, logging, "
                    "and infrastructure development. Indigenous communities have traditionally been the "
                    "forest's most effective protectors, with their territories showing significantly "
                    "lower deforestation rates."
                ),
                question="What are the main causes of Amazon deforestation?",
                expected_answer="cattle ranching, soy cultivation, logging, and infrastructure development",
                ground_truth_facts=[
                    "main drivers of deforestation include cattle ranching, soy cultivation, logging, and infrastructure development",
                    "deforestation rates have been alarming",
                    "11,568 square kilometers were cleared in 2022"
                ],
                irrelevant_facts=[
                    "spans across nine countries",
                    "Brazil containing approximately 60%",
                    "5.5 million square kilometers",
                    "10% of all known species",
                    "Amazon River",
                    "world's largest river by volume",
                    "2.2 billion tons of carbon dioxide annually",
                    "Indigenous communities"
                ],
                description="Environmental issue requiring identification of multiple causal factors",
                domain="geography"
            ),
            
            # Sports
            TestExample(
                id="sports_tennis_1",
                context=(
                    "Wimbledon Championships 2023:\n"
                    "The prestigious Wimbledon tennis tournament concluded yesterday with thrilling finals "
                    "at the All England Club. In the men's singles final, Novak Djokovic defeated Carlos "
                    "Alcaraz in a five-set marathon that lasted 4 hours and 42 minutes, with scores of "
                    "1-6, 7-6, 6-1, 3-6, 6-4. This victory marked Djokovic's 7th Wimbledon title and "
                    "23rd Grand Slam overall, tying him with Serena Williams for the most Grand Slam "
                    "titles in the Open Era. The women's singles final saw Marketa Vondrousova upset "
                    "Ons Jabeur 6-4, 6-4 to claim her first Grand Slam title. The match was played on "
                    "Centre Court in front of 15,000 spectators and millions of viewers worldwide. "
                    "Prize money for the singles champions was £2.35 million each."
                ),
                question="How long did the men's final match last?",
                expected_answer="4 hours and 42 minutes",
                ground_truth_facts=[
                    "lasted 4 hours and 42 minutes",
                    "five-set marathon",
                    "Novak Djokovic defeated Carlos Alcaraz"
                ],
                irrelevant_facts=[
                    "All England Club",
                    "scores of 1-6, 7-6, 6-1, 3-6, 6-4",
                    "7th Wimbledon title",
                    "23rd Grand Slam overall",
                    "Marketa Vondrousova",
                    "Ons Jabeur 6-4, 6-4",
                    "Centre Court",
                    "15,000 spectators",
                    "£2.35 million"
                ],
                description="Sports reporting requiring specific time duration extraction",
                domain="sports"
            ),
            
            # Psychology/Health
            TestExample(
                id="psychology_sleep_1",
                context=(
                    "Sleep Research Findings:\n"
                    "A groundbreaking study published in the Journal of Sleep Medicine reveals significant "
                    "insights about sleep patterns and cognitive performance. Researchers at Stanford "
                    "University followed 500 participants over six months, monitoring their sleep quality "
                    "using advanced EEG technology and assessing cognitive function through daily tests. "
                    "The study found that individuals who maintained consistent sleep schedules, going to "
                    "bed and waking up at the same time every day, showed 23% better performance on memory "
                    "tasks compared to those with irregular sleep patterns. Additionally, participants who "
                    "got 7-9 hours of sleep per night demonstrated superior problem-solving abilities and "
                    "faster reaction times. Dr. Emily Chen, the lead researcher, noted that sleep quality "
                    "was more important than sleep quantity, with REM sleep phases being particularly "
                    "crucial for memory consolidation. The study also revealed that blue light exposure "
                    "from screens within two hours of bedtime reduced sleep quality by an average of 15%."
                ),
                question="What sleep factors improved memory task performance by 23%?",
                expected_answer="consistent sleep schedules",
                ground_truth_facts=[
                    "individuals who maintained consistent sleep schedules",
                    "going to bed and waking up at the same time every day",
                    "showed 23% better performance on memory tasks"
                ],
                irrelevant_facts=[
                    "Journal of Sleep Medicine",
                    "Stanford University",
                    "500 participants over six months",
                    "advanced EEG technology",
                    "7-9 hours of sleep per night",
                    "Dr. Emily Chen",
                    "REM sleep phases",
                    "blue light exposure from screens"
                ],
                description="Scientific study requiring identification of specific performance factors",
                domain="psychology"
            ),
            
            # Food/Culture
            TestExample(
                id="culture_cuisine_1",
                context=(
                    "Mediterranean Diet and Health Benefits:\n"
                    "The Mediterranean diet, traditional to countries bordering the Mediterranean Sea "
                    "such as Greece, Italy, and Spain, has gained worldwide recognition for its health "
                    "benefits. This dietary pattern emphasizes the consumption of olive oil as the primary "
                    "source of fat, along with high amounts of fruits, vegetables, whole grains, legumes, "
                    "nuts, and fish. Red meat consumption is limited to a few times per month, while "
                    "moderate wine consumption, particularly red wine with meals, is common. A landmark "
                    "study published in the New England Journal of Medicine followed 7,447 participants "
                    "for five years and found that those following the Mediterranean diet had a 30% "
                    "lower risk of cardiovascular disease compared to those on a low-fat diet. The diet "
                    "is also associated with reduced inflammation, better brain health, and longevity. "
                    "Researchers attribute these benefits primarily to the high content of omega-3 fatty "
                    "acids from fish and nuts, antioxidants from fruits and vegetables, and monounsaturated "
                    "fats from olive oil."
                ),
                question="What are the primary sources of the Mediterranean diet's health benefits?",
                expected_answer="omega-3 fatty acids, antioxidants, and monounsaturated fats",
                ground_truth_facts=[
                    "omega-3 fatty acids from fish and nuts",
                    "antioxidants from fruits and vegetables",
                    "monounsaturated fats from olive oil",
                    "Researchers attribute these benefits primarily to"
                ],
                irrelevant_facts=[
                    "countries bordering the Mediterranean Sea",
                    "Greece, Italy, and Spain",
                    "olive oil as the primary source of fat",
                    "Red meat consumption is limited",
                    "moderate wine consumption",
                    "New England Journal of Medicine",
                    "7,447 participants for five years",
                    "30% lower risk of cardiovascular disease"
                ],
                description="Nutritional science requiring identification of causal mechanisms",
                domain="culture"
            ),
            
            # Space/Astronomy  
            TestExample(
                id="astronomy_mars_1",
                context=(
                    "Mars Exploration Mission Update:\n"
                    "NASA's Perseverance rover has made a remarkable discovery on Mars, finding organic "
                    "molecules in rock samples from the Jezero Crater. The rover, which landed on Mars "
                    "in February 2021, has been exploring the ancient river delta region using its "
                    "sophisticated suite of scientific instruments including the PIXL spectrometer and "
                    "SUPERCAM laser. The organic compounds were detected in sedimentary rocks that are "
                    "approximately 3.5 billion years old, formed when water was present on the Martian "
                    "surface. Dr. Michael Meyer, lead scientist for NASA's Mars Exploration Program, "
                    "emphasized that while organic molecules can be produced by non-biological processes, "
                    "their presence in these ancient rocks significantly increases the possibility that "
                    "microbial life once existed on Mars. The samples will be stored in sealed containers "
                    "for a future Mars Sample Return mission planned for the 2030s, which will bring "
                    "them back to Earth for detailed laboratory analysis."
                ),
                question="Where were the organic molecules discovered on Mars?",
                expected_answer="Jezero Crater sedimentary rocks",
                ground_truth_facts=[
                    "organic molecules in rock samples from the Jezero Crater",
                    "detected in sedimentary rocks",
                    "ancient river delta region"
                ],
                irrelevant_facts=[
                    "Perseverance rover",
                    "landed on Mars in February 2021",
                    "PIXL spectrometer and SUPERCAM laser",
                    "approximately 3.5 billion years old",
                    "Dr. Michael Meyer",
                    "Mars Exploration Program",
                    "non-biological processes",
                    "Mars Sample Return mission",
                    "planned for the 2030s"
                ],
                description="Space exploration requiring location identification from technical description",
                domain="astronomy"
            ),
            
            # Architecture/Engineering
            TestExample(
                id="engineering_bridge_1",
                context=(
                    "Golden Gate Bridge Engineering Marvel:\n"
                    "The Golden Gate Bridge, spanning the Golden Gate strait between San Francisco and "
                    "Marin County, is considered one of the greatest engineering achievements of the 20th "
                    "century. Construction began in 1933 under the supervision of chief engineer Joseph "
                    "Strauss, with the bridge officially opening to traffic on May 27, 1937. The suspension "
                    "bridge design features two main towers reaching 746 feet above the water, connected "
                    "by main cables that are each composed of 27,572 individual steel wires. The total "
                    "length of the bridge is 8,980 feet, with the main span measuring 4,200 feet, making "
                    "it the longest suspension bridge span in the world at the time of completion. The "
                    "distinctive International Orange color was chosen not only for aesthetic reasons but "
                    "also to enhance visibility in San Francisco's frequent fog. During construction, "
                    "innovative safety measures included the use of a safety net that saved 19 workers' "
                    "lives, earning them the nickname 'Halfway to Hell Club.'"
                ),
                question="What safety innovation was used during the bridge's construction?",
                expected_answer="safety net",
                ground_truth_facts=[
                    "innovative safety measures included the use of a safety net",
                    "saved 19 workers' lives",
                    "nicknamed 'Halfway to Hell Club'"
                ],
                irrelevant_facts=[
                    "Golden Gate strait",
                    "San Francisco and Marin County",
                    "Construction began in 1933",
                    "chief engineer Joseph Strauss",
                    "May 27, 1937",
                    "suspension bridge design",
                    "746 feet above the water",
                    "27,572 individual steel wires",
                    "International Orange color"
                ],
                description="Engineering history requiring identification of specific innovation",
                domain="engineering"
            ),
            
            # Politics/Government
            TestExample(
                id="politics_constitution_1",
                context=(
                    "US Constitutional Amendment Process:\n"
                    "The United States Constitution provides two methods for proposing amendments and "
                    "two methods for ratifying them, creating four possible paths for constitutional "
                    "change. The most common method requires a two-thirds majority vote in both the "
                    "House of Representatives and the Senate to propose an amendment, followed by "
                    "ratification by three-fourths of state legislatures. An alternative ratification "
                    "method involves approval by special conventions in three-fourths of the states, "
                    "which was used only once for the 21st Amendment that repealed Prohibition. The "
                    "second proposal method allows two-thirds of state legislatures to call for a "
                    "national constitutional convention, though this has never been successfully used. "
                    "Of the 27 amendments currently in the Constitution, the first 10, known as the "
                    "Bill of Rights, were ratified simultaneously in 1791. The most recent amendment, "
                    "the 27th Amendment concerning congressional pay, was ratified in 1992 after lying "
                    "dormant for over 200 years since its original proposal in 1789."
                ),
                question="Which amendment used the alternative ratification method involving state conventions?",
                expected_answer="21st Amendment",
                ground_truth_facts=[
                    "21st Amendment that repealed Prohibition",
                    "approval by special conventions in three-fourths of the states",
                    "used only once"
                ],
                irrelevant_facts=[
                    "two methods for proposing amendments",
                    "two-thirds majority vote",
                    "House of Representatives and the Senate",
                    "three-fourths of state legislatures",
                    "national constitutional convention",
                    "27 amendments currently",
                    "Bill of Rights",
                    "ratified simultaneously in 1791",
                    "27th Amendment",
                    "ratified in 1992"
                ],
                description="Constitutional law requiring identification of specific procedural example",
                domain="politics"
            ),
            
            # Environment/Conservation
            TestExample(
                id="environment_coral_1",
                context=(
                    "Great Barrier Reef Conservation Efforts:\n"
                    "The Great Barrier Reef, located off the coast of Queensland, Australia, is the "
                    "world's largest coral reef system, stretching over 2,300 kilometers and visible "
                    "from space. This UNESCO World Heritage site supports over 65,000 jobs in tourism "
                    "and fishing industries and contributes approximately $6.4 billion annually to the "
                    "Australian economy. However, the reef faces unprecedented threats from climate "
                    "change, with rising ocean temperatures causing massive coral bleaching events. "
                    "The most severe bleaching occurred in 2016 and 2017, affecting over 50% of shallow "
                    "water corals. In response, the Australian government launched the Reef 2050 Plan, "
                    "a comprehensive conservation strategy that includes water quality improvements, "
                    "crown-of-thorns starfish control, and climate change mitigation efforts. Scientists "
                    "are also developing innovative solutions such as coral gardening, where heat-resistant "
                    "coral varieties are cultivated in nurseries and then transplanted to damaged reef "
                    "areas. The Marine Park Authority estimates that without immediate action, 99% of "
                    "the reef could be lost by 2050."
                ),
                question="What innovative conservation technique involves cultivating heat-resistant corals?",
                expected_answer="coral gardening",
                ground_truth_facts=[
                    "coral gardening",
                    "heat-resistant coral varieties are cultivated in nurseries",
                    "transplanted to damaged reef areas"
                ],
                irrelevant_facts=[
                    "coast of Queensland, Australia",
                    "world's largest coral reef system",
                    "stretching over 2,300 kilometers",
                    "UNESCO World Heritage site",
                    "65,000 jobs in tourism",
                    "$6.4 billion annually",
                    "massive coral bleaching events",
                    "2016 and 2017",
                    "Reef 2050 Plan",
                    "Marine Park Authority"
                ],
                description="Environmental conservation requiring identification of specific technique",
                domain="environment"
            ),
            
            # Art/Music
            TestExample(
                id="art_renaissance_1",
                context=(
                    "Renaissance Art and Patronage:\n"
                    "During the Italian Renaissance, wealthy merchant families and the Catholic Church "
                    "served as the primary patrons of art, commissioning works from renowned artists "
                    "like Leonardo da Vinci, Michelangelo, and Raphael. The Medici family of Florence "
                    "was particularly influential, supporting artists financially and providing them "
                    "with workshops and materials. Their patronage system allowed artists to focus on "
                    "their craft without worrying about basic survival needs. This period saw revolutionary "
                    "developments in artistic techniques, including the mastery of linear perspective "
                    "by Brunelleschi, the use of chiaroscuro (light and shadow) by Caravaggio, and the "
                    "development of oil painting techniques that allowed for greater detail and color "
                    "depth. The Vatican commissioned Michelangelo to paint the Sistine Chapel ceiling "
                    "between 1508 and 1512, a work that took four years to complete and required the "
                    "artist to work lying on his back on scaffolding. This masterpiece depicts scenes "
                    "from the Book of Genesis and is considered one of the greatest achievements in "
                    "Western art."
                ),
                question="How long did Michelangelo take to complete the Sistine Chapel ceiling?",
                expected_answer="four years",
                ground_truth_facts=[
                    "between 1508 and 1512",
                    "took four years to complete",
                    "Michelangelo to paint the Sistine Chapel ceiling"
                ],
                irrelevant_facts=[
                    "Italian Renaissance",
                    "wealthy merchant families",
                    "Catholic Church",
                    "Leonardo da Vinci, Michelangelo, and Raphael",
                    "Medici family of Florence",
                    "linear perspective by Brunelleschi",
                    "chiaroscuro (light and shadow) by Caravaggio",
                    "oil painting techniques",
                    "scenes from the Book of Genesis"
                ],
                description="Art history requiring temporal duration calculation",
                domain="art"
            ),
            
            # Philosophy/Ethics
            TestExample(
                id="philosophy_ethics_1",
                context=(
                    "AI Ethics and Decision Making:\n"
                    "The development of artificial intelligence systems capable of making autonomous "
                    "decisions has raised significant ethical concerns about accountability and bias. "
                    "A recent controversy emerged when a hiring algorithm used by a major tech company "
                    "was found to discriminate against female candidates for software engineering positions. "
                    "The algorithm had been trained on historical hiring data that reflected past biases "
                    "in the industry, leading it to perpetuate these inequalities. Dr. Rebecca Williams, "
                    "an AI ethics researcher at MIT, argues that machine learning systems inherit the "
                    "biases present in their training data, making it crucial to address these issues "
                    "during the development phase. The concept of algorithmic fairness has emerged as "
                    "a key principle, requiring that AI systems treat all individuals equitably regardless "
                    "of protected characteristics like gender, race, or age. Companies are now implementing "
                    "bias testing protocols and diverse review boards to evaluate their AI systems before "
                    "deployment. The European Union has proposed comprehensive AI regulations that would "
                    "require transparency in algorithmic decision-making and mandate human oversight for "
                    "high-risk applications."
                ),
                question="What causes AI systems to perpetuate discrimination according to Dr. Williams?",
                expected_answer="biases in training data",
                ground_truth_facts=[
                    "machine learning systems inherit the biases present in their training data",
                    "trained on historical hiring data that reflected past biases",
                    "Dr. Rebecca Williams"
                ],
                irrelevant_facts=[
                    "autonomous decisions",
                    "major tech company",
                    "female candidates for software engineering",
                    "AI ethics researcher at MIT",
                    "algorithmic fairness",
                    "protected characteristics",
                    "bias testing protocols",
                    "European Union",
                    "human oversight"
                ],
                description="Ethics discussion requiring identification of causal explanation",
                domain="philosophy"
            )
        ]
        
        print(f"📊 Created {len(self.test_examples)} test examples covering domains:")
        domains = set(example.domain for example in self.test_examples)
        for domain in sorted(domains):
            count = sum(1 for ex in self.test_examples if ex.domain == domain)
            print(f"   • {domain}: {count} examples")
        
    def evaluate_example(self, example: TestExample) -> AttentionTestResult:
        """Evaluate attention-based fact retrieval on a single example"""
        print(f"\n🔍 Testing example: {example.id} ({example.domain})")
        
        try:
            # Use the retriever to extract facts
            retrieval_result = self.retriever.retrieve_facts(
                context=example.context,
                question=example.question, 
                cot_response=example.expected_answer,
                use_cross_evaluation=True
            )
            
            retrieved_facts = retrieval_result.get('retrieved_facts', [])
            
            # Calculate metrics
            precision, recall, f1 = self.calculate_precision_recall_f1(
                retrieved_facts, example.ground_truth_facts, example.irrelevant_facts
            )
            
            top_k_acc = self.calculate_top_k_accuracy(
                retrieved_facts[:self.k], example.ground_truth_facts
            )
            
            return AttentionTestResult(
                example_id=example.id,
                retrieved_facts=retrieved_facts,
                ground_truth_facts=example.ground_truth_facts,
                irrelevant_facts=example.irrelevant_facts,
                precision=precision,
                recall=recall,
                f1_score=f1,
                top_k_accuracy=top_k_acc,
                attention_scores=retrieval_result
            )
            
        except Exception as e:
            print(f"❌ Error evaluating example {example.id}: {e}")
            return AttentionTestResult(
                example_id=example.id,
                retrieved_facts=[],
                ground_truth_facts=example.ground_truth_facts,
                irrelevant_facts=example.irrelevant_facts,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                top_k_accuracy=0.0,
                attention_scores={}
            )
    
    def calculate_precision_recall_f1(self, retrieved_facts, ground_truth_facts, irrelevant_facts):
        """Calculate precision, recall, and F1 score"""
        if not retrieved_facts:
            return 0.0, 0.0, 0.0
            
        # Extract text from retrieved facts
        retrieved_texts = [fact.get('text', '') for fact in retrieved_facts]
        
        # Count matches with ground truth
        true_positives = 0
        false_positives = 0
        
        for retrieved_text in retrieved_texts:
            if any(self.text_similarity(retrieved_text, gt_fact) > 0.5 for gt_fact in ground_truth_facts):
                true_positives += 1
            else:
                false_positives += 1
        
        # Calculate metrics
        precision = true_positives / len(retrieved_texts) if retrieved_texts else 0.0
        recall = true_positives / len(ground_truth_facts) if ground_truth_facts else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def calculate_top_k_accuracy(self, top_k_facts, ground_truth_facts):
        """Calculate how many ground truth facts are in top K"""
        if not ground_truth_facts:
            return 0.0
            
        top_k_texts = [fact.get('text', '') for fact in top_k_facts]
        
        found_count = 0
        for gt_fact in ground_truth_facts:
            if any(self.text_similarity(gt_fact, top_k_text) > 0.5 for top_k_text in top_k_texts):
                found_count += 1
                
        return found_count / len(ground_truth_facts)
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def run_all_tests(self):
        """Run tests on all examples"""
        print("🚀 Starting General English Attention Module Testing")
        print("=" * 60)
        
        if not ATTENTION_VIZ_AVAILABLE:
            raise RuntimeError("attention_viz not available")
            
        # Setup
        self.setup_model()
        self.create_test_dataset()
        
        # Run tests
        self.results = []
        for example in self.test_examples:
            result = self.evaluate_example(example)
            self.results.append(result)
            
        # Generate summary
        self.generate_summary()
        self.save_results()
        
    def generate_summary(self):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 GENERAL ENGLISH ATTENTION MODULE TEST SUMMARY")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results to analyze")
            return
            
        # Calculate aggregate metrics
        avg_precision = np.mean([r.precision for r in self.results])
        avg_recall = np.mean([r.recall for r in self.results])
        avg_f1 = np.mean([r.f1_score for r in self.results])
        avg_top_k_acc = np.mean([r.top_k_accuracy for r in self.results])
        
        print(f"📈 Overall Performance (K={self.k}):")
        print(f"   Average Precision: {avg_precision:.3f}")
        print(f"   Average Recall: {avg_recall:.3f}")
        print(f"   Average F1 Score: {avg_f1:.3f}")
        print(f"   Average Top-{self.k} Accuracy: {avg_top_k_acc:.3f}")
        
        # Performance by domain
        print(f"\n📋 Performance by Domain:")
        domains = set(example.domain for example in self.test_examples)
        for domain in sorted(domains):
            domain_results = [r for r, ex in zip(self.results, self.test_examples) if ex.domain == domain]
            if domain_results:
                domain_f1 = np.mean([r.f1_score for r in domain_results])
                domain_precision = np.mean([r.precision for r in domain_results])
                domain_recall = np.mean([r.recall for r in domain_results])
                print(f"   • {domain}: F1={domain_f1:.3f}, P={domain_precision:.3f}, R={domain_recall:.3f}")
        
        # Best and worst performers
        sorted_results = sorted(zip(self.results, self.test_examples), 
                              key=lambda x: x[0].f1_score, reverse=True)
        
        print(f"\n🏆 Best Performers:")
        for i, (result, example) in enumerate(sorted_results[:3]):
            print(f"   {i+1}. {result.example_id} ({example.domain}): F1={result.f1_score:.3f}")
            
        print(f"\n⚠️  Worst Performers:")
        for i, (result, example) in enumerate(sorted_results[-3:]):
            print(f"   {i+1}. {result.example_id} ({example.domain}): F1={result.f1_score:.3f}")
            
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        if avg_f1 >= 0.7:
            print("   ✅ EXCELLENT: Attention module working very well on general English text")
        elif avg_f1 >= 0.5:
            print("   🟡 GOOD: Attention module working reasonably well on general English text")
        elif avg_f1 >= 0.3:
            print("   🟠 MODERATE: Attention module needs improvement for general English text")
        else:
            print("   ❌ POOR: Attention module needs significant work for general English text")
            
    def save_results(self):
        """Save detailed results to files"""
        output_dir = Path("general_english_attention_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_data = []
        for result, example in zip(self.results, self.test_examples):
            results_data.append({
                'example_id': result.example_id,
                'domain': example.domain,
                'description': example.description,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'top_k_accuracy': result.top_k_accuracy,
                'retrieved_facts': result.retrieved_facts,
                'ground_truth_facts': result.ground_truth_facts,
                'irrelevant_facts': result.irrelevant_facts
            })
            
        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
            
        # Save summary statistics
        summary = {
            'test_config': {
                'model_name': self.model_name,
                'k': self.k,
                'num_examples': len(self.test_examples)
            },
            'aggregate_metrics': {
                'avg_precision': float(np.mean([r.precision for r in self.results])),
                'avg_recall': float(np.mean([r.recall for r in self.results])),
                'avg_f1': float(np.mean([r.f1_score for r in self.results])),
                'avg_top_k_accuracy': float(np.mean([r.top_k_accuracy for r in self.results]))
            },
            'domain_breakdown': {},
            'per_example_results': [
                {
                    'example_id': r.example_id,
                    'domain': ex.domain,
                    'precision': r.precision,
                    'recall': r.recall,
                    'f1_score': r.f1_score,
                    'top_k_accuracy': r.top_k_accuracy
                }
                for r, ex in zip(self.results, self.test_examples)
            ]
        }
        
        # Calculate domain breakdown
        domains = set(example.domain for example in self.test_examples)
        for domain in domains:
            domain_results = [r for r, ex in zip(self.results, self.test_examples) if ex.domain == domain]
            if domain_results:
                summary['domain_breakdown'][domain] = {
                    'avg_precision': float(np.mean([r.precision for r in domain_results])),
                    'avg_recall': float(np.mean([r.recall for r in domain_results])),
                    'avg_f1': float(np.mean([r.f1_score for r in domain_results])),
                    'avg_top_k_accuracy': float(np.mean([r.top_k_accuracy for r in domain_results])),
                    'count': len(domain_results)
                }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"   - Detailed results: {output_dir}/detailed_results.json")
        print(f"   - Summary: {output_dir}/summary.json")

def main():
    """Main function to run the general English attention module test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test attention module on general English text")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model to test")
    parser.add_argument("--k", type=int, default=5, help="Number of top facts to retrieve")
    parser.add_argument("--examples", type=int, default=15, help="Number of test examples to run")
    
    args = parser.parse_args()
    
    # Create and run test suite
    test_suite = GeneralEnglishAttentionTestSuite(model_name=args.model, k=args.k)
    
    try:
        test_suite.run_all_tests()
        print("\n🎉 General English attention module testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 