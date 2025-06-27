from typing import Any, Text, Dict, List, Optional
import time
import pickle
import nltk
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import  FollowupAction
import logging
import os
import google.generativeai as genai
import json
import requests
import re
import traceback
import pandas as pd
import numpy as np
import faiss

from rasa_sdk.events import SlotSet

# Ensure NLTK resources are downloaded (do it once at server start)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# mistralchatbot-458317
def call_gemini_api(prompt: str) -> str:
    """
    Calls the Gemini API with the provided prompt using Google AI SDK.
    Returns "VALID" or "INVALID" based on the model's response.
    """
    logger.info("Calling Gemini API with prompt: %s", prompt)

    # Configure the Gemini API with your API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Gemini API key not found in environment variables.")
        return "INVALID"

    genai.configure(api_key=api_key)

    # Initialize the model
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

    try:
        # Generate content
        response = model.generate_content(prompt)

        logger.info("Received response from Gemini API")

        # Extract the text from the response
        generated_text = ""
        if hasattr(response, 'text') and response.text:
            generated_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            logger.error("Could not extract text from Gemini API response.")
            return "INVALID"

        logger.info("Generated text: %s", generated_text)

        # Check if the response contains VALID (case-insensitive)
        if "VALID" in generated_text.upper():
            return "VALID"
        else:
            return "INVALID"
    except Exception as e:
        logger.exception("Exception occurred during Gemini API call: %s", e)
        return "INVALID"


class ActionValidateProjectDescription(Action):
    def name(self) -> Text:
        return "action_validate_project_description"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        description = tracker.get_slot("description")
        logger.info("Validating project description: %s", description)

        if not description:
            dispatcher.utter_message(text="Veuillez fournir une description détaillée du projet.")
            logger.warning("No project description provided by the user (in validation).")
            return [SlotSet("description", None)]

        prompt = (
            f"Veuillez évaluer si la description suivante du projet est suffisamment détaillée et pertinente. "
            f"Elle doit inclure des informations sur les objectifs, le contexte ou les défis du projet. \n"
            f"Description: '{description}'.\n"
            "Répondez uniquement avec 'VALID' si la description est acceptable ou 'INVALID' sinon."
        )
        logger.info("Constructed prompt for description validation: %s", prompt)

        result = call_gemini_api(prompt)
        logger.info("Gemini API validation result for description: %s", result)

        if result.strip().upper() == "VALID":
            logger.info("Project description validated as VALID.")
            return []
        else:
            dispatcher.utter_message(response="utter_insufficient_description")
            logger.info("Project description validated as INVALID. Prompting user for a more detailed description.")
            return [SlotSet("description", None)]


class ActionValidateProjectTasks(Action):
    def name(self) -> Text:
        return "action_validate_project_tasks"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        tasks = tracker.get_slot("tasks")
        logger.info("Validating project tasks: %s", tasks)

        if not tasks:
            dispatcher.utter_message(text="Merci de fournir une liste détaillée des tâches du projet.")
            logger.warning("No project tasks provided by the user (in validation).")
            return [SlotSet("tasks", None)]

        prompt = (
            f"Veuillez évaluer si la description suivante des tâches du projet est suffisamment détaillée et pertinente. "
            f"Elle devrait inclure l'objectif, les ressources, les dépendances ou les résultats attendus pour chaque tâche clé.\n"
            f"Tâches: '{tasks}'.\n"
            "Répondez uniquement par 'VALID' si la description est acceptable ou 'INVALID' sinon."
        )
        logger.info("Constructed prompt for tasks validation: %s", prompt)

        result = call_gemini_api(prompt)
        logger.info("Gemini API validation result for tasks: %s", result)

        if result.strip().upper() == "VALID":
            logger.info("Project tasks validated as VALID.")
            return []
        else:
            dispatcher.utter_message(response="utter_insufficient_tasks")
            logger.info("Project tasks validated as INVALID. Prompting user for more detailed tasks.")
            return [SlotSet("tasks", None)]


class ActionCheckProjectSlots(Action):
    def name(self) -> Text:
        return "action_check_project_slots"

    async def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get latest message intent
        latest_message = tracker.latest_message
        intent_name = latest_message.get('intent', {}).get('name', '')

        # Log the latest message for debugging
        logger.info(f"Latest message: {latest_message}")
        logger.info(f"Intent: {intent_name}")

        # Define required slots and their user-friendly names
        required_slots = [
            "project_name",
            "description",
            "duration",
            "complexity",
            "sector",
            "tasks"
        ]

        # More user-friendly names for the prompt
        slot_prompts_fr = {
            "project_name": "le nom du projet",
            "description": "une description détaillée",
            "duration": "la durée prévue (en mois)",
            "complexity": "le niveau de complexité (sur une échelle de 1 à 5)",
            "sector": "le secteur ou l'industrie",
            "tasks": "les tâches principales"
        }

        # Check for extracted entities and set slots before validation
        slot_events = self._extract_and_set_slots(tracker, latest_message)

        # Apply any slot events from extracted entities immediately
        # This is crucial to update the tracker's state with the extracted values
        for event in slot_events:
            # FIX: Access dictionary keys correctly instead of treating event as an object
            tracker.slots[event['name']] = event['value']

        # Normalize duration to keep only numeric value
        duration_raw = tracker.get_slot("duration")
        if duration_raw is not None:
            match = re.search(r"(\d+\.?\d*)", str(duration_raw))
            if match:
                duration_val = float(match.group(1))
                logger.info(f"Normalizing duration '{duration_raw}' to {duration_val}")
                slot_events.append(SlotSet("duration", duration_val))
                tracker.slots["duration"] = duration_val
            else:
                logger.info(f"Could not parse numeric duration from '{duration_raw}'")

        # Check complexity validity first if it exists
        complexity = tracker.get_slot("complexity")
        if complexity is not None:
            try:
                complexity_float = float(complexity)
                if not (1 <= complexity_float <= 5):
                    logger.info(f"Complexity value {complexity} is outside the valid range (1-5). Resetting.")
                    dispatcher.utter_message(response="utter_invalid_complexity")
                    slot_events.append(SlotSet("complexity", None))
                    # Update tracker's state
                    tracker.slots["complexity"] = None
            except ValueError:
                logger.info(f"Complexity value {complexity} is not a valid number. Resetting.")
                dispatcher.utter_message(response="utter_invalid_complexity")
                slot_events.append(SlotSet("complexity", None))
                # Update tracker's state
                tracker.slots["complexity"] = None

        # Log current slot values (after applying entity extractions)
        logger.info("Current slot values (after entity extraction):")
        for slot_name in required_slots:
            logger.info(f"  {slot_name}: {tracker.get_slot(slot_name)}")

        # Identify missing slots
        missing_slots = []
        for slot_name in required_slots:
            # Re-check complexity here in case it was just reset
            if slot_name == "complexity" and tracker.get_slot("complexity") is None:
                if slot_name not in missing_slots:  # Avoid duplicates if reset above
                    missing_slots.append(slot_name)
            elif tracker.get_slot(slot_name) is None:
                missing_slots.append(slot_name)

        logger.info(f"Missing slots: {missing_slots}")

        if not missing_slots:
            logger.info("All required project slots appear to be filled.")
            # Set flag to indicate completion, triggering exit from the loop
            slot_events.append(SlotSet("all_slots_filled", True))
            return slot_events
        else:
            # Construct the prompt dynamically in French
            prompt_parts = [slot_prompts_fr.get(slot, slot) for slot in missing_slots]

            if len(prompt_parts) == 1:
                prompt = f"Pourriez-vous me fournir {prompt_parts[0]}?"
            elif len(prompt_parts) == 2:
                prompt = f"Pourriez-vous me fournir {prompt_parts[0]} et {prompt_parts[1]}?"
            else:
                # For more than 2, list them with commas and 'et' before the last one
                all_but_last = ", ".join(prompt_parts[:-1])
                last = prompt_parts[-1]
                prompt = f"Pourriez-vous me fournir {all_but_last}, et {last}?"

            logger.info(f"Sending prompt: {prompt}")
            dispatcher.utter_message(text=prompt)

            # Ensure the flag is false so the loop continues
            slot_events.append(SlotSet("all_slots_filled", False))
            return slot_events

    def _extract_and_set_slots(self, tracker, latest_message):
        """
        Extract slot values from the latest message entities and intent.
        This is particularly useful for CALM's LLM-based entity extraction.
        """
        # Log the entities for debugging
        entities = latest_message.get('entities', [])
        logger.info(f"Extracted entities: {entities}")

        # Check for commands in the message (CALM LLM entity extraction)
        commands = latest_message.get('commands', [])
        logger.info(f"Commands from LLM: {commands}")

        # Process extracted slots from commands
        slot_events = []
        for command in commands:
            if command.get('command') == 'set slot':
                slot_name = command.get('name')
                value = command.get('value')

                # Log raw slot name for debugging
                logger.info(f"Raw slot name from LLM: {slot_name}")

                # Comprehensive mapping between various LLM command names and our actual slot names
                # Include all possible variations the LLM might use
                slot_mapping = {
                    # Project name variations
                    'project_name': 'project_name',
                    'nom_projet': 'project_name',
                    'nom_du_projet': 'project_name',
                    'nom': 'project_name',

                    # Duration variations
                    'duration': 'duration',
                    'project_duration': 'duration',
                    'project_duration_months': 'duration',
                    'duree': 'duration',
                    'duree_projet': 'duration',
                    'duree_du_projet': 'duration',
                    'periode': 'duration',

                    # Complexity variations
                    'complexity': 'complexity',
                    'project_complexity': 'complexity',
                    'complexite': 'complexity',
                    'niveau_complexite': 'complexity',
                    'niveau_de_complexite': 'complexity',

                    # Sector variations
                    'sector': 'sector',
                    'project_sector': 'sector',
                    'secteur': 'sector',
                    'industrie': 'sector',
                    'domaine': 'sector',

                    # Description variations
                    'description': 'description',
                    'project_description': 'description',
                    'description_projet': 'description',
                    'description_du_projet': 'description',

                    # Tasks variations
                    'tasks': 'tasks',
                    'project_tasks': 'tasks',
                    'taches': 'tasks',
                    'taches_projet': 'tasks',
                    'taches_du_projet': 'tasks',

                    # CALM specific prefixes - keep these as well
                    'collect_project_info_project_name': 'project_name',
                    'collect_project_info_description': 'description',
                    'collect_project_info_duration': 'duration',
                    'collect_project_info_complexity': 'complexity',
                    'collect_project_info_sector': 'sector',
                    'collect_project_info_tasks': 'tasks',
                }

                # Try to get the actual slot name from our mapping, or use a fuzzy match
                actual_slot = self._get_best_slot_match(slot_name, slot_mapping)

                if actual_slot in tracker.slots:
                    logger.info(f"Setting slot from command: {actual_slot} = {value}")
                    slot_events.append(SlotSet(actual_slot, value))
                else:
                    logger.warning(f"Unknown slot name from command: {slot_name} -> {actual_slot}")

        return slot_events

    def _get_best_slot_match(self, slot_name, slot_mapping):
        """
        Get the best match for a slot name, even handling partial matches or spelling variations.
        """
        # Direct lookup first (exact match)
        if slot_name in slot_mapping:
            return slot_mapping[slot_name]

        # Convert to lowercase for case-insensitive matching
        slot_name_lower = slot_name.lower()
        if slot_name_lower in slot_mapping:
            return slot_mapping[slot_name_lower]

        # Check if the slot name contains any of our known keys
        for key, value in slot_mapping.items():
            # If the slot name contains a key as a substring
            if key.lower() in slot_name_lower:
                logger.info(f"Fuzzy matched '{slot_name}' to '{key}' -> '{value}'")
                return value

        # Additional logic for specific types (you can expand this as needed)
        if "duration" in slot_name_lower or "duree" in slot_name_lower or "months" in slot_name_lower or "mois" in slot_name_lower:
            return "duration"
        elif "name" in slot_name_lower or "nom" in slot_name_lower:
            return "project_name"
        elif "complex" in slot_name_lower:
            return "complexity"
        elif "sector" in slot_name_lower or "secteur" in slot_name_lower or "industry" in slot_name_lower:
            return "sector"
        elif "task" in slot_name_lower or "tache" in slot_name_lower:
            return "tasks"
        elif "desc" in slot_name_lower:
            return "description"

        # If all else fails, return the original name
        return slot_name


class ActionGetProjectRecommendations(Action):
    def name(self) -> Text:
        return "action_get_project_recommendations"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # First, gather all project details from slots
        project_name = tracker.get_slot("project_name")
        description = tracker.get_slot("description")
        duration = tracker.get_slot("duration")
        complexity = tracker.get_slot("complexity")
        sector = tracker.get_slot("sector")
        tasks = tracker.get_slot("tasks")

        logger.info("Preparing to get recommendations for project: %s", project_name)

        # Prepare data for the endpoint
        project_data = {
            "inputs": {
                "Nom du projet": project_name,
                "Description": description,
                "Durée (mois)": float(duration),
                "Complexité (1-5)": float(complexity),
                "Secteur": sector,
                "Tâches Identifiées": tasks
            }
        }

        logger.info("Calling recommendation endpoint with data: %s", json.dumps(project_data))

        try:
            # Get API credentials from environment variables
            hf_token = os.getenv("HF_TOKEN")
            endpoint_url = os.getenv("ENDPOINT_URL")

            if not hf_token or not endpoint_url:
                logger.error("HF_TOKEN or ENDPOINT_URL environment variables not set")
                dispatcher.utter_message(
                    text="Je ne peux pas accéder au service de recommandations en ce moment. Veuillez réessayer plus tard.")
                return []

            # Setup headers
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            }

            # Call the endpoint
            response = requests.post(endpoint_url, headers=headers, json=project_data, timeout=240)

            # Check if the request was successful
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info("Received response from endpoint: %s", json.dumps(result))

                    # Extract the generated text
                    if "generated_text" in result:
                        generated_text = result["generated_text"]

                        # Try to parse as JSON
                        try:
                            json_output = json.loads(generated_text)
                            logger.info("Successfully parsed response as JSON")

                            # Validate recommendations with Gemini
                            validated_recommendations = self.validate_recommendations(json_output,
                                                                                      project_data["inputs"])

                            # Store the recommendations as a JSON string in the slot
                            return [
                                SlotSet("recommendations", json.dumps(validated_recommendations, ensure_ascii=False))]

                        except json.JSONDecodeError:
                            logger.warning("Generated text is not valid JSON: %s", generated_text)
                            # Process raw text with Gemini anyway
                            validated_recommendations = self.validate_recommendations({"raw_text": generated_text},
                                                                                      project_data["inputs"])
                            return [
                                SlotSet("recommendations", json.dumps(validated_recommendations, ensure_ascii=False))]
                    else:
                        logger.error("No 'generated_text' in response: %s", json.dumps(result))
                        dispatcher.utter_message(
                            text="Je n'ai pas pu générer de recommandations pour votre projet. Le format de réponse est incorrect.")
                        return []

                except Exception as e:
                    logger.exception("Error processing response: %s", e)
                    dispatcher.utter_message(text="Une erreur s'est produite lors du traitement des recommandations.")
                    return []
            else:
                logger.error("Error response from endpoint: %s - %s", response.status_code, response.text)
                dispatcher.utter_message(
                    text="Je n'ai pas pu accéder au service de recommandations en ce moment. Veuillez réessayer plus tard.")
                return []

        except Exception as e:
            logger.exception("Exception during recommendation API call: %s", e)
            dispatcher.utter_message(text="Une erreur s'est produite lors de la génération des recommandations.")
            return []

    def validate_recommendations(self, recommendations, project_info):
        """
        Validate and improve recommendations using Gemini API
        """
        logger.info("Validating recommendations with Gemini API")

        try:
            # Configure the Gemini API with API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("Gemini API key not found in environment variables.")
                return recommendations

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

            # Prepare the prompt for validation with explicit guidance on required format
            prompt = f"""
            J'ai un projet avec les informations suivantes:
            - Nom du projet: {project_info['Nom du projet']}
            - Description: {project_info['Description']}
            - Durée: {project_info['Durée (mois)']} mois
            - Complexité: {project_info['Complexité (1-5)']}
            - Secteur: {project_info['Secteur']}
            - Tâches identifiées: {project_info['Tâches Identifiées']}

            Un système a généré les recommandations suivantes pour ce projet:
            {json.dumps(recommendations, ensure_ascii=False, indent=2)}

            Veuillez analyser ces recommandations et les transformer en un JSON structuré avec les sections suivantes:

            1. "Ressources Recommandées": qui contiendra deux sous-sections:
               - "Ressources Humaines": liste détaillée des profils nécessaires, chacun avec:
                 * "Role": titre du poste
                 * "Description": description des responsabilités
                 * "Estimation Quantité": nombre estimé de personnes requises (un chiffre précis, pas une fourchette)

               - "Ressources Matérielles": liste des équipements et outils nécessaires

            2. "Employés Alloués": nombre total d'employés recommandés pour le projet
               IMPORTANT: Évaluez de façon critique le nombre d'employés proposé par le système. 
               S'il vous semble incorrect par rapport à la complexité et la durée du projet, 
               ou s'il ne correspond pas à la somme des "Estimation Quantité" dans les ressources humaines, 
               corrigez-le selon votre jugement. Donnez une estimation réaliste.

            3. "Répartition par Compétences": un objet avec les rôles comme clés et le nombre d'employés comme valeurs.
               IMPORTANT: Ajustez cette répartition pour qu'elle soit logique et corresponde au nombre total d'employés 
               que vous avez déterminé dans "Employés Alloués". La répartition par compétences doit refléter 
               les besoins réels du projet.

            4. "Conseils Généraux": conseils généraux pour la réussite du projet

            CRITÈRES OBLIGATOIRES:
            - Les sections "Employés Alloués" et "Répartition par Compétences" doivent être présentes et cohérentes entre elles
            - Le nombre total dans "Employés Alloués" DOIT correspondre EXACTEMENT à la somme des valeurs dans "Répartition par Compétences"
            - Les rôles dans "Répartition par Compétences" doivent correspondre aux rôles listés dans "Ressources Humaines"
            - Chaque rôle doit avoir une description détaillée et une estimation de quantité précise (pas de fourchettes comme "1 à 3")
            - Vos recommandations doivent être adaptées à la complexité du projet (échelle 1-5): {project_info['Complexité (1-5)']}
            - Vos recommandations doivent être adaptées à la durée du projet: {project_info['Durée (mois)']} mois

            Répondez uniquement avec le JSON corrigé et amélioré, sans autre texte.
            """

            # Call Gemini API with temperature to encourage critical evaluation
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Lower temperature for more consistent critical analysis
                )
            )

            # Extract the text from the response
            generated_text = ""
            if hasattr(response, 'text') and response.text:
                generated_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                logger.error("Could not extract text from Gemini API response.")
                return recommendations

            logger.info("Gemini validation response received")

            # Try to parse the response as JSON
            try:
                # First, try to extract JSON if it's wrapped in code blocks
                if "```json" in generated_text and "```" in generated_text.split("```json", 1)[1]:
                    json_text = generated_text.split("```json", 1)[1].split("```", 1)[0].strip()
                    validated_recommendations = json.loads(json_text)
                elif "```" in generated_text and "```" in generated_text.split("```", 1)[1]:
                    json_text = generated_text.split("```", 1)[1].split("```", 1)[0].strip()
                    validated_recommendations = json.loads(json_text)
                else:
                    validated_recommendations = json.loads(generated_text)

                logger.info("Successfully parsed Gemini response as JSON")

                # Verify required fields are present
                if "Employés Alloués" not in validated_recommendations:
                    logger.warning("Missing 'Employés Alloués' in Gemini response, adding default value")
                    validated_recommendations["Employés Alloués"] = self._calculate_total_employees(
                        validated_recommendations)

                if "Répartition par Compétences" not in validated_recommendations:
                    logger.warning("Missing 'Répartition par Compétences' in Gemini response, adding default value")
                    validated_recommendations["Répartition par Compétences"] = self._create_competence_distribution(
                        validated_recommendations)

                return validated_recommendations

            except json.JSONDecodeError:
                logger.warning("Could not parse Gemini response as JSON: %s", generated_text)
                return recommendations  # Return original if can't parse Gemini response

        except Exception as e:
            logger.exception("Exception during Gemini validation: %s", e)
            return recommendations  # Return original recommendations if validation fails

    def _calculate_total_employees(self, recommendations):
        """Calculate total employees based on resources if not provided"""
        total = 0

        if "Ressources Recommandées" in recommendations and "Ressources Humaines" in recommendations[
            "Ressources Recommandées"]:
            resources = recommendations["Ressources Recommandées"]["Ressources Humaines"]
            if isinstance(resources, list):
                for resource in resources:
                    if isinstance(resource, dict) and "Estimation Quantité" in resource:
                        # Try to convert to numeric
                        try:
                            quantity = resource["Estimation Quantité"]
                            if isinstance(quantity, str):
                                # Handle case where quantity might be a range or have text
                                if "-" in quantity:
                                    # Use the higher end of a range
                                    quantity = quantity.split("-")[1].strip()

                                # Extract just the numeric part
                                import re
                                numeric_match = re.search(r'(\d+(\.\d+)?)', quantity)
                                if numeric_match:
                                    quantity = numeric_match.group(1)

                            total += float(quantity)
                        except (ValueError, TypeError):
                            # Default to 1 if we can't parse
                            total += 1

        return int(total) if total > 0 else 3  # Default to 3 if no valid calculation

    def _create_competence_distribution(self, recommendations):
        """Create competence distribution based on resources if not provided"""
        distribution = {}

        if "Ressources Recommandées" in recommendations and "Ressources Humaines" in recommendations[
            "Ressources Recommandées"]:
            resources = recommendations["Ressources Recommandées"]["Ressources Humaines"]
            if isinstance(resources, list):
                for resource in resources:
                    if isinstance(resource, dict) and "Role" in resource and "Estimation Quantité" in resource:
                        role = resource["Role"]
                        try:
                            quantity = resource["Estimation Quantité"]
                            if isinstance(quantity, str):
                                # Handle case where quantity might be a range or have text
                                if "-" in quantity:
                                    # Use the higher end of a range
                                    quantity = quantity.split("-")[1].strip()

                                # Extract just the numeric part
                                import re
                                numeric_match = re.search(r'(\d+(\.\d+)?)', quantity)
                                if numeric_match:
                                    quantity = numeric_match.group(1)

                            distribution[role] = int(float(quantity))
                        except (ValueError, TypeError):
                            # Default to 1 if we can't parse
                            distribution[role] = 1

        return distribution if distribution else {"Chef de Projet": 1,
                                                  "Développeur": 2}  # Default values if no valid calculation


class ActionPresentRecommendations(Action):
    def name(self) -> Text:
        return "action_present_recommendations"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get recommendations from slot (now always stored as a JSON string)
        recommendations_str = tracker.get_slot("recommendations")

        if not recommendations_str:
            dispatcher.utter_message(
                text="Je n'ai pas encore de recommandations pour votre projet. Veuillez d'abord demander des recommandations.")
            return []

        logger.info("Presenting recommendations to user")

        try:
            # Parse the JSON string back to a Python object
            recommendations = json.loads(recommendations_str)
            logger.info(f"Successfully parsed recommendations from slot: {type(recommendations)}")

            # Check if we have a raw_text field (non-JSON response from endpoint)
            if recommendations.get("raw_text"):
                dispatcher.utter_message(
                    text=f"Voici mes recommandations pour votre projet:\n\n{recommendations['raw_text']}")
                return []

            # Format the structured recommendations
            message = "📊 **Voici mes recommandations pour votre projet** 📊\n\n"

            # Add recommended resources
            if "Ressources Recommandées" in recommendations:
                message += "👥 **RESSOURCES RECOMMANDÉES**\n"
                resources = recommendations["Ressources Recommandées"]

                # Handle Human Resources
                if isinstance(resources, dict) and "Ressources Humaines" in resources:
                    message += "**Ressources Humaines**:\n"
                    human_resources = resources["Ressources Humaines"]
                    if isinstance(human_resources, list):
                        for resource in human_resources:
                            if isinstance(resource, dict):
                                message += f"* **{resource.get('Role', 'N/A')}**: {resource.get('Description', 'N/A')} (Quantité: {resource.get('Estimation Quantité', 'N/A')})\n"
                            else:
                                message += f"* {resource}\n"
                    else:
                        message += f"* {human_resources}\n"

                # Handle Material Resources
                if isinstance(resources, dict) and "Ressources Matérielles" in resources:
                    message += "**Ressources Matérielles**:\n"
                    material_resources = resources["Ressources Matérielles"]
                    if isinstance(material_resources, list):
                        for resource in material_resources:
                            message += f"* {resource}\n"
                    else:
                        message += f"* {material_resources}\n"

                message += "\n"

            # Add Employees Allocated
            if "Employés Alloués" in recommendations:
                message += f"👥 **EMPLOYÉS ALLOUÉS**: {recommendations['Employés Alloués']}\n\n"

            # Add Competence Distribution
            if "Répartition par Compétences" in recommendations:
                message += "📊 **RÉPARTITION PAR COMPÉTENCES**:\n"
                distribution = recommendations["Répartition par Compétences"]
                if isinstance(distribution, dict):
                    for role, count in distribution.items():
                        message += f"* {role}: {count}\n"
                else:
                    message += f"{distribution}\n"

                message += "\n"

            # Add General Advice
            if "Conseils Généraux" in recommendations:
                message += "💡 **CONSEILS GÉNÉRAUX**:\n"
                advice = recommendations["Conseils Généraux"]
                if isinstance(advice, list):
                    for i, tip in enumerate(advice, 1):
                        message += f"{i}. {tip}\n"
                else:
                    message += f"{advice}\n"

            # Send the formatted message
            dispatcher.utter_message(text=message)
            return []

        except Exception as e:
            logger.exception("Error presenting recommendations: %s", e)
            # Fallback to basic display if there's an error
            try:
                # Since recommendations_str is already a JSON string, we can display it directly
                dispatcher.utter_message(text=f"Voici mes recommandations pour votre projet:\n\n{recommendations_str}")
            except:
                dispatcher.utter_message(text="Je n'ai pas pu formater les recommandations correctement.")
            return []


class ActionHandleRecommendationsFollowup(Action):
    def name(self) -> Text:
        return "action_handle_recommendations_followup"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get the latest message text
        latest_message = tracker.latest_message.get("text", "")
        logger.info(f"Processing follow-up question: {latest_message}")

        # Get recommendations from slot
        recommendations_str = tracker.get_slot("recommendations")
        if not recommendations_str:
            dispatcher.utter_message(
                text="Je n'ai pas de recommandations à expliquer. Veuillez d'abord demander des recommandations.")
            return []

        try:
            # Parse the JSON string back to a Python object
            recommendations = json.loads(recommendations_str)

            # Process the query using Gemini
            response = self.process_query_with_gemini(latest_message, recommendations)

            # Send the response
            dispatcher.utter_message(text=response)

            # Check if this was a modification request and update the recommendations if needed
            updated_recommendations = self.check_for_modifications(latest_message, recommendations, response)
            if updated_recommendations != recommendations:
                logger.info("Recommendations were updated based on user request")
                return [SlotSet("recommendations", json.dumps(updated_recommendations, ensure_ascii=False))]

            return []

        except Exception as e:
            logger.exception(f"Error handling follow-up question: {e}")
            dispatcher.utter_message(
                text="Je suis désolé, je n'ai pas pu traiter votre question sur les recommandations.")
            return []

    def process_query_with_gemini(self, query: str, recommendations: Dict) -> str:
        """Process the user query about recommendations using Gemini API"""
        logger.info("Processing query with Gemini API")

        try:
            # Configure the Gemini API with API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("Gemini API key not found in environment variables.")
                return "Je ne peux pas traiter votre question pour le moment."

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

            # Prepare the prompt for answering follow-up questions
            prompt = f"""
            Un utilisateur a reçu les recommandations suivantes pour son projet:
            {json.dumps(recommendations, ensure_ascii=False, indent=2)}

            L'utilisateur pose maintenant la question ou fait la demande suivante:
            "{query}"

            Veuillez répondre à cette question ou demande de manière précise et professionnelle.

            Si l'utilisateur demande une explication sur une partie spécifique (comme "Employés Alloués", 
            "Répartition par Compétences", ou un rôle spécifique), expliquez le raisonnement derrière cette recommandation.

            Si l'utilisateur suggère une modification (comme augmenter/diminuer le nombre d'employés, 
            ajouter/supprimer des compétences, etc.), indiquez si cette modification est raisonnable 
            et quelles seraient les implications de ce changement.

            Votre réponse doit être UNIQUEMENT en français et adaptée au contexte professionnel.
            """

            # Call Gemini API
            response = model.generate_content(prompt)

            # Extract the text from the response
            generated_text = ""
            if hasattr(response, 'text') and response.text:
                generated_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                logger.error("Could not extract text from Gemini API response.")
                return "Je ne peux pas traiter votre question pour le moment."

            return generated_text

        except Exception as e:
            logger.exception(f"Exception during Gemini query processing: {e}")
            return "Je suis désolé, je n'ai pas pu traiter votre question."

    def check_for_modifications(self, query: str, recommendations: Dict, gemini_response: str) -> Dict:
        """Check if the user is requesting a modification and update recommendations if needed"""
        try:
            # Configure the Gemini API with API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("Gemini API key not found in environment variables.")
                return recommendations

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

            # Prepare the prompt to check for and implement modifications
            prompt = f"""
            Un utilisateur a reçu les recommandations suivantes pour son projet:
            {json.dumps(recommendations, ensure_ascii=False, indent=2)}

            L'utilisateur a fait la demande suivante:
            "{query}"

            Votre réponse précédente était:
            "{gemini_response}"

            TÂCHE: Déterminez si l'utilisateur demande une modification des recommandations. 
            Si OUI, modifiez le JSON des recommandations en conséquence.
            Si NON, retournez EXACTEMENT le même JSON sans aucune modification.

            Types de modifications possibles:
            1. Changer le nombre total d'"Employés Alloués"
            2. Ajouter/supprimer/modifier un rôle dans "Ressources Humaines"
            3. Modifier la "Répartition par Compétences"
            4. Ajouter/supprimer/modifier les "Ressources Matérielles"
            5. Modifier les "Conseils Généraux"

            RÈGLES IMPORTANTES:
            - Si vous modifiez le nombre total d'"Employés Alloués", assurez-vous que la somme des valeurs dans "Répartition par Compétences" correspond exactement à ce nouveau nombre.
            - Si vous modifiez "Répartition par Compétences", assurez-vous que le total correspond à "Employés Alloués".
            - Si vous ajoutez/supprimez un rôle dans "Ressources Humaines", mettez également à jour "Répartition par Compétences".
            - Maintenez la cohérence entre toutes les sections des recommandations.

            Répondez uniquement avec le JSON mis à jour ou inchangé, sans autre texte.
            """

            # Call Gemini API
            response = model.generate_content(prompt)

            # Extract the text from the response
            generated_text = ""
            if hasattr(response, 'text') and response.text:
                generated_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                logger.error("Could not extract text from Gemini API response.")
                return recommendations

            # Try to parse the response as JSON
            try:
                # First, try to extract JSON if it's wrapped in code blocks
                if "```json" in generated_text and "```" in generated_text.split("```json", 1)[1]:
                    json_text = generated_text.split("```json", 1)[1].split("```", 1)[0].strip()
                    updated_recommendations = json.loads(json_text)
                elif "```" in generated_text and "```" in generated_text.split("```", 1)[1]:
                    json_text = generated_text.split("```", 1)[1].split("```", 1)[0].strip()
                    updated_recommendations = json.loads(json_text)
                else:
                    updated_recommendations = json.loads(generated_text)

                logger.info("Successfully parsed modified recommendations")
                return updated_recommendations
            except json.JSONDecodeError:
                logger.warning("Could not parse modification response as JSON")
                return recommendations



        except Exception as e:
            logger.exception(f"Exception during modification check: {e}")
            return recommendations


class ActionConfirmResourceValues(Action):
    """
    Confirms the current resource values from recommendations or updates them if needed
    """

    def name(self) -> Text:
        return "action_confirm_resource_values"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get recommendations from slot
        recommendations_str = tracker.get_slot("recommendations")
        if not recommendations_str:
            dispatcher.utter_message(
                text="Je n'ai pas de recommandations sur lesquelles baser l'estimation de coût. Veuillez d'abord obtenir des recommandations.")
            return [SlotSet("resources_confirmed", False)]

        try:
            # Parse the recommendations
            recommendations = json.loads(recommendations_str)

            # Extract skills distribution and total employees
            skills_distribution = {}
            total_employees = 0

            # Extract skills distribution if available
            if "Répartition par Compétences" in recommendations:
                skills_distribution = recommendations["Répartition par Compétences"]

            # Extract total employees if available, or calculate from skills
            if "Employés Alloués" in recommendations:
                try:
                    total_employees = int(recommendations["Employés Alloués"])
                except (ValueError, TypeError):
                    # If we can't parse, sum up the skills
                    total_employees = sum(skills_distribution.values()) if skills_distribution else 0
            else:
                total_employees = sum(skills_distribution.values()) if skills_distribution else 0

            # Save these as slots for the cost estimation flow
            events = [
                SlotSet("resources_confirmed", True),
                SlotSet("skills_distribution", json.dumps(skills_distribution, ensure_ascii=False)),
                SlotSet("total_employees", total_employees)
            ]

            # Prepare a message to inform user about the current resources
            message = (
                "📊 **Ressources actuelles du projet** 📊\n\n"
                f"Selon les recommandations, nous avons la répartition suivante:\n"
                f"- Nombre total d'employés: {total_employees}\n"
                "- Répartition des compétences:\n"
            )

            # Add skills distribution details
            for skill, count in skills_distribution.items():
                message += f"  • {skill}: {count} personnes\n"

            message += "\nCes informations sont-elles correctes pour l'estimation des coûts?"

            dispatcher.utter_message(text=message)
            return events

        except Exception as e:
            logger.exception(f"Error confirming resource values: {e}")
            dispatcher.utter_message(
                text="Je n'ai pas pu récupérer les informations sur les ressources. Veuillez vérifier que les recommandations sont valides.")
            return [SlotSet("resources_confirmed", False)]


class ActionProcessResourceUpdate(Action):
    """
    Process updates to the resource allocation based on user input
    """

    def name(self) -> Text:
        return "action_process_resource_update"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get the latest message text
        latest_message = tracker.latest_message.get("text", "")

        # Get current skills distribution
        skills_distribution_str = tracker.get_slot("skills_distribution")
        if not skills_distribution_str:
            dispatcher.utter_message(
                text="Je ne trouve pas les informations sur la répartition des compétences. Veuillez recommencer.")
            return [SlotSet("resources_confirmed", False)]

        try:
            # Parse the skills distribution
            skills_distribution = json.loads(skills_distribution_str)

            # Use Gemini to process the resource update request
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("Gemini API key not found in environment variables.")
                dispatcher.utter_message(
                    text="Je ne peux pas traiter votre demande de mise à jour pour le moment. Veuillez réessayer plus tard.")
                return [SlotSet("resources_confirmed", False)]

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

            # Prepare the prompt for processing resource updates
            prompt = f"""
            Voici la répartition actuelle des compétences pour un projet:
            {json.dumps(skills_distribution, ensure_ascii=False, indent=2)}

            L'utilisateur souhaite apporter des modifications et a dit:
            "{latest_message}"

            TÂCHE: Analysez la demande de l'utilisateur et mettez à jour la répartition des compétences.

            Types de modifications possibles:
            1. Ajout d'une nouvelle compétence
            2. Suppression d'une compétence existante
            3. Modification du nombre de personnes pour une compétence

            Répondez avec un JSON contenant:
            1. La répartition mise à jour
            2. Une description des modifications apportées
            3. Le nombre total d'employés après modifications

            Format de réponse EXACT:
            {{
                "updated_skills": {{...}},
                "changes_description": "...",
                "total_employees": X
            }}
            """

            # Call the API
            response = model.generate_content(prompt)

            # Extract the response
            generated_text = ""
            if hasattr(response, 'text') and response.text:
                generated_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                logger.error("Could not extract text from Gemini API response.")
                dispatcher.utter_message(
                    text="Je ne peux pas traiter votre demande de mise à jour pour le moment. Veuillez réessayer plus tard.")
                return [SlotSet("resources_confirmed", False)]

            # Parse the JSON response
            # First, try to extract JSON if it's wrapped in a code block
            if "```json" in generated_text and "```" in generated_text.split("```json", 1)[1]:
                json_text = generated_text.split("```json", 1)[1].split("```", 1)[0].strip()
                update_result = json.loads(json_text)
            elif "```" in generated_text and "```" in generated_text.split("```", 1)[1]:
                json_text = generated_text.split("```", 1)[1].split("```", 1)[0].strip()
                update_result = json.loads(json_text)
            else:
                # Try parsing the whole text
                update_result = json.loads(generated_text)

            # Update the skills distribution and total employees
            updated_skills = update_result.get("updated_skills", {})
            changes_description = update_result.get("changes_description", "")
            total_employees = update_result.get("total_employees", sum(updated_skills.values()))

            # Send confirmation message to the user
            message = (
                "✅ **Mise à jour des ressources** ✅\n\n"
                f"{changes_description}\n\n"
                f"Nouvelle répartition des compétences:\n"
            )

            # Add updated skills distribution details
            for skill, count in updated_skills.items():
                message += f"  • {skill}: {count} personnes\n"

            message += f"\nNombre total d'employés: {total_employees}"

            dispatcher.utter_message(text=message)

            # Save the updated values
            return [
                SlotSet("skills_distribution", json.dumps(updated_skills, ensure_ascii=False)),
                SlotSet("total_employees", total_employees),
                SlotSet("resources_confirmed", True)
            ]

        except Exception as e:
            logger.exception(f"Error processing resource update: {e}")
            dispatcher.utter_message(
                text="Je n'ai pas pu mettre à jour les ressources. Veuillez vérifier votre demande et réessayer.")
            return [SlotSet("resources_confirmed", False)]


class ActionAskForSalaries(Action):
    """
    Ask for all salaries at once using an optimized approach
    """

    def name(self) -> Text:
        return "action_ask_for_salaries"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get the skills distribution
        skills_distribution_str = tracker.get_slot("skills_distribution")
        if not skills_distribution_str:
            dispatcher.utter_message(
                text="Je ne trouve pas les informations sur la répartition des compétences. Veuillez revenir à l'étape précédente.")
            return []

        try:
            # Parse the skills distribution
            skills_distribution = json.loads(skills_distribution_str)

            # Prepare the message asking for all salaries and FG% at once
            message = (
                "💰 **Informations financières pour l'estimation** 💰\n\n"
                f"Pour calculer le coût total du projet sur {tracker.get_slot('duration')} mois, j'ai besoin des informations suivantes en une seule réponse:\n\n"
                "1️⃣ **Salaires mensuels bruts** pour chaque compétence:\n"
            )

            # List all skills with specific prompting for each
            for i, skill in enumerate(skills_distribution.keys(), 1):
                message += f"   - Combien gagne un(e) {skill} par mois? (en €)\n"

            message += (
                "\n2️⃣ **Pourcentage de frais généraux (FG%)** à appliquer pour couvrir les coûts indirects\n"
                "   (locaux, équipement, administration, etc.)\n\n"
                "Exemple de réponse: \"Un Chef de projet gagne 4500€/mois, un Développeur 3800€, "
                "un Designer 3500€, et le FG est de 20%.\"\n\n"
                "Je traiterai toutes ces informations ensemble pour calculer l'estimation globale du projet."
            )

            dispatcher.utter_message(text=message)

            # Initialize the salaries slot as an empty object
            return [SlotSet("salaries", json.dumps({}, ensure_ascii=False))]

        except Exception as e:
            logger.exception(f"Error asking for salaries: {e}")
            dispatcher.utter_message(
                text="Une erreur s'est produite lors de la demande des informations financières. Veuillez réessayer.")
            return []


class ActionCollectSalaryInfo(Action):
    """
    Amélioré pour collecter correctement les informations de salaire d'une seule entrée utilisateur,
    avec une gestion améliorée de l'extraction et des champs manquants
    """

    def name(self) -> Text:
        return "action_collect_salary_info"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get the latest message text
        latest_message = tracker.latest_message.get("text", "")

        # Get the skills distribution
        skills_distribution_str = tracker.get_slot("skills_distribution")
        if not skills_distribution_str:
            dispatcher.utter_message(
                text="Je ne trouve pas les informations sur la répartition des compétences. Veuillez revenir à l'étape précédente.")
            return []

        # Récupérer les salaires existants avec une meilleure gestion des erreurs
        salaries_str = tracker.get_slot("salaries")
        salaries = {}
        if salaries_str:
            try:
                salaries = json.loads(salaries_str)
            except (json.JSONDecodeError, TypeError):
                logger.error(f"Failed to parse salaries: {salaries_str}")
                salaries = {}

        # Récupérer le FG% existant si disponible
        existing_fg = tracker.get_slot("fg_percentage")

        try:
            # Parse the skills distribution
            skills_distribution = json.loads(skills_distribution_str)

            # Use Gemini to extract salaries and FG% from the message
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("Gemini API key not found in environment variables.")
                dispatcher.utter_message(
                    text="Je ne peux pas traiter votre demande pour le moment. Veuillez réessayer plus tard.")
                return []

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

            # Préparer le prompt en incluant les données déjà collectées
            prompt = f"""
            L'utilisateur doit fournir les salaires mensuels pour les compétences suivantes et le pourcentage de frais généraux (FG%).

            Compétences: {json.dumps(list(skills_distribution.keys()), ensure_ascii=False)}

            Informations déjà collectées:
            - Salaires existants: {json.dumps(salaries, ensure_ascii=False) if salaries else "Aucun"}
            - FG% existant: {existing_fg if existing_fg is not None else "Non défini"}

            L'utilisateur a répondu:
            "{latest_message}"

            TÂCHE: Extrayez tous les salaires mentionnés et le pourcentage de frais généraux (FG%).
            IMPORTANT: Ne réinitialisez PAS les données déjà collectées! Fusionnez les nouvelles données avec les données existantes.

            Règles d'extraction:
            1. Associez chaque compétence à un montant numérique en euros si présent
            2. Ignorez les unités (€, euros) et ne gardez que les valeurs numériques
            3. Pour le FG%, extrayez uniquement la valeur numérique du pourcentage
            4. Utilisez la correspondance la plus proche si les noms de compétences ne correspondent pas exactement
            5. Gérez différentes formulations de salaires (ex: "gagne 3000€", "salaire de 3000€", "3000 par mois", etc.)
            6. IMPORTANT: Ne considérez pas comme manquants les éléments déjà collectés!

            Format de réponse EXACT:
            {{
                "extracted_salaries": {{
                    "compétence1": valeur_numerique,
                    "compétence2": valeur_numerique,
                    ...
                }},
                "fg_percentage": valeur_numerique_ou_null,
                "missing_info": {{
                    "skills": ["compétence3", "compétence4", ...],
                    "fg_percentage": true_ou_false
                }}
            }}
            """

            # Appeler l'API
            response = model.generate_content(prompt)

            # Extraire la réponse
            generated_text = ""
            if hasattr(response, 'text') and response.text:
                generated_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                logger.error("Could not extract text from Gemini API response.")
                dispatcher.utter_message(
                    text="Je ne peux pas traiter votre demande pour le moment. Veuillez réessayer plus tard.")
                return []

            # Parse the JSON response with error handling
            try:
                # Try to extract JSON if it's wrapped in a code block
                if "```json" in generated_text and "```" in generated_text.split("```json", 1)[1]:
                    json_text = generated_text.split("```json", 1)[1].split("```", 1)[0].strip()
                    extraction_result = json.loads(json_text)
                elif "```" in generated_text and "```" in generated_text.split("```", 1)[1]:
                    json_text = generated_text.split("```", 1)[1].split("```", 1)[0].strip()
                    extraction_result = json.loads(json_text)
                else:
                    # Try parsing the whole text
                    extraction_result = json.loads(generated_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini response: {e}")
                extraction_result = {
                    "extracted_salaries": {},
                    "fg_percentage": None,
                    "missing_info": {
                        "skills": list(skills_distribution.keys()),
                        "fg_percentage": existing_fg is None
                    }
                }

            # Récupérer les salaires extraits, FG%, et infos manquantes
            extracted_salaries = extraction_result.get("extracted_salaries", {})
            fg_percentage = extraction_result.get("fg_percentage")

            # Fusionner les salaires existants avec les nouveaux
            updated_salaries = {**salaries, **extracted_salaries}  # Donne priorité aux nouvelles valeurs

            # Journal de débogage
            logger.debug(f"Previous salaries: {salaries}")
            logger.debug(f"Extracted salaries: {extracted_salaries}")
            logger.debug(f"Updated salaries: {updated_salaries}")

            # Identifier les compétences toujours manquantes après mise à jour
            missing_skills = [skill for skill in skills_distribution.keys() if skill not in updated_salaries]

            # Vérifier si FG% est défini dans ce message ou précédemment
            fg_percentage_final = fg_percentage if fg_percentage is not None else existing_fg
            missing_fg = fg_percentage_final is None

            # Events à retourner
            events = []

            # Définir FG percentage si fourni
            if fg_percentage is not None:
                events.append(SlotSet("fg_percentage", fg_percentage))

            # Vérifier si nous avons toutes les informations requises
            all_salaries_collected = len(missing_skills) == 0 and not missing_fg

            # TOUJOURS mettre à jour les salaires collectés jusqu'à présent, même partiellement
            events.append(SlotSet("salaries", json.dumps(updated_salaries, ensure_ascii=False)))

            if all_salaries_collected:
                # Nous avons toutes les informations requises, confirmer et passer à l'étape suivante
                confirmation_message = "✅ **Informations financières collectées** ✅\n\nJ'ai enregistré les données suivantes:\n\n**Salaires mensuels:**\n"
                for role, salary in updated_salaries.items():
                    confirmation_message += f"- {role}: {salary}€/mois\n"

                confirmation_message += f"\n**Frais généraux (FG%)**: {fg_percentage_final}%"

                dispatcher.utter_message(text=confirmation_message)
                events.append(SlotSet("all_salaries_collected", True))
            else:
                # Il manque encore des informations
                missing_message = "📝 **Informations partiellement enregistrées** 📝\n\n"

                # D'abord reconnaître ce que nous avons reçu
                missing_message += "J'ai bien enregistré les informations suivantes:\n"
                if updated_salaries:
                    missing_message += "\n**Salaires mensuels:**\n"
                    for role, salary in updated_salaries.items():
                        missing_message += f"- {role}: {salary}€/mois\n"

                if fg_percentage_final is not None:
                    missing_message += f"\n**Frais généraux (FG%)**: {fg_percentage_final}%\n"

                # Puis demander les informations manquantes
                missing_message += "\nCependant, il me manque encore:\n"

                # Lister les compétences manquantes
                if missing_skills:
                    missing_message += "\n**Salaires mensuels pour:**\n"
                    for skill in missing_skills:
                        missing_message += f"- {skill}\n"

                # Demander FG% si manquant
                if missing_fg:
                    missing_message += "\n**Le pourcentage de frais généraux (FG%)**\n"

                missing_message += "\nPouvez-vous me fournir ces informations manquantes?\n"

                dispatcher.utter_message(text=missing_message)
                events.append(SlotSet("all_salaries_collected", False))

            return events

        except Exception as e:
            logger.exception(f"Error collecting salary info: {e}")
            dispatcher.utter_message(
                text="Une erreur s'est produite lors de la collecte des informations financières. Veuillez réessayer avec un format plus simple, par exemple: \"Un chef de projet gagne 4500€/mois, le FG est de 20%\"")
            return []


class ActionProcessFGPercentage(Action):
    """
    Process FG percentage if it wasn't provided with salaries
    """

    def name(self) -> Text:
        return "action_process_fg_percentage"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Check if we already have the FG percentage
        fg_percentage = tracker.get_slot("fg_percentage")
        if fg_percentage is not None:
            # We already have it, move to calculation
            return [FollowupAction("action_calculate_cost_estimation")]

        # Get the latest message text
        latest_message = tracker.latest_message.get("text", "")

        # Extract the percentage from the message
        percentage = self._extract_percentage_from_message(latest_message)

        if percentage is None:
            dispatcher.utter_message(
                text="Je n'ai pas pu déterminer le pourcentage de frais généraux. Veuillez indiquer un nombre (ex: 20 pour 20%)."
            )
            return []

        # Save the percentage
        dispatcher.utter_message(text=f"Merci! J'ai enregistré le pourcentage de frais généraux: {percentage}%.")

        # Move to calculation
        return [
            SlotSet("fg_percentage", percentage),
            FollowupAction("action_calculate_cost_estimation")
        ]

    def _extract_percentage_from_message(self, message: str) -> Optional[float]:
        """
        Extract percentage from user message
        """
        # Patterns to match percentages
        patterns = [
            r'(\d+(?:[.,]\d+)?)\s*%',  # Matches "20%" or "20 %"
            r'(\d+(?:[.,]\d+)?)\s*(?:pour ?cent|pourcent)',  # Matches "20 pourcent" or "20 pour cent"
            r'(\d+(?:[.,]\d+)?)'  # Matches just a number
        ]

        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            if matches:
                # Take the first match
                percentage_str = matches[0]
                if isinstance(percentage_str, tuple):  # In case there are multiple capture groups
                    percentage_str = next((s for s in percentage_str if s), '')
                # Replace comma with decimal point
                percentage_str = percentage_str.replace(',', '.')

                try:
                    return float(percentage_str)
                except ValueError:
                    continue

        return None


class ActionCalculateCostEstimation(Action):
    """
    Calculate the total project cost based on all collected information
    """

    def name(self) -> Text:
        return "action_calculate_cost_estimation"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get all the required information
        skills_distribution_str = tracker.get_slot("skills_distribution")
        salaries_str = tracker.get_slot("salaries")
        fg_percentage = tracker.get_slot("fg_percentage")
        project_duration = tracker.get_slot("duration")  # From the original project information

        # Verify we have all necessary information
        missing_info = []
        if not skills_distribution_str:
            missing_info.append("la répartition des compétences")
        if not salaries_str:
            missing_info.append("les salaires")
        if fg_percentage is None:
            missing_info.append("le pourcentage de frais généraux")
        if not project_duration:
            missing_info.append("la durée du projet")

        if missing_info:
            missing_str = ", ".join(missing_info)
            dispatcher.utter_message(
                text=f"Je ne peux pas calculer l'estimation de coût sans {missing_str}. Veuillez fournir ces informations."
            )
            return []

        try:
            # Parse the information
            skills_distribution = json.loads(skills_distribution_str)
            salaries = json.loads(salaries_str)
            fg_percentage = float(fg_percentage)
            project_duration = float(project_duration)

            # Calculate the base cost (employee salaries)
            total_salary_cost = 0
            skill_costs = {}

            for skill, count in skills_distribution.items():
                if skill in salaries:
                    salary = float(salaries[skill])
                    skill_cost = count * salary * project_duration
                    skill_costs[skill] = {
                        "count": count,
                        "salary": salary,
                        "total_cost": skill_cost
                    }
                    total_salary_cost += skill_cost
                else:
                    # Should not happen as we check for all salaries, but just in case
                    dispatcher.utter_message(
                        text=f"Je n'ai pas trouvé de salaire pour '{skill}'. Veuillez fournir cette information."
                    )
                    return []

            # Calculate the overhead cost
            overhead_cost = total_salary_cost * (fg_percentage / 100)

            # Calculate the total cost
            total_project_cost = total_salary_cost + overhead_cost

            # Prepare the result
            result = {
                "total_salary_cost": total_salary_cost,
                "overhead_cost": overhead_cost,
                "total_project_cost": total_project_cost,
                "skill_costs": skill_costs,
                "fg_percentage": fg_percentage,
                "project_duration": project_duration
            }

            # Format the message to display the result
            message = self._format_cost_result(result)

            # Send the message
            dispatcher.utter_message(text=message)

            # Save the result
            return [
                SlotSet("cost_estimation", json.dumps(result, ensure_ascii=False)),
                SlotSet("cost_estimation_complete", True)
            ]

        except Exception as e:
            logger.exception(f"Error calculating cost estimation: {e}")
            dispatcher.utter_message(
                text="Une erreur s'est produite lors du calcul de l'estimation de coût. Veuillez vérifier les données et réessayer."
            )
            return []

    def _format_cost_result(self, result: Dict) -> str:
        """
        Format the cost estimation result for display
        """
        message = "💰 **ESTIMATION DU COÛT DU PROJET** 💰\n\n"

        # Add the project duration
        message += f"**Durée du projet**: {result['project_duration']} mois\n\n"

        # Add the skill costs
        message += "**Coûts par compétence**:\n"
        for skill, data in result['skill_costs'].items():
            skill_cost = f"{data['total_cost']:,.2f}€".replace(",", " ")
            message += f"- {skill} ({data['count']} personnes à {data['salary']:,.2f}€/mois): {skill_cost}\n"

        message += f"\n**Coût total des salaires**: {result['total_salary_cost']:,.2f}€".replace(",", " ")

        # Add the overhead
        message += f"\n**Frais généraux ({result['fg_percentage']}%)**: {result['overhead_cost']:,.2f}€".replace(",",
                                                                                                                 " ")

        # Add the total cost
        message += f"\n\n**COÛT TOTAL ESTIMÉ DU PROJET**: {result['total_project_cost']:,.2f}€".replace(",", " ")

        message += "\n\nCette estimation est basée sur les informations que vous m'avez fournies. Vous pouvez me demander d'expliquer certains aspects ou suggérer des ajustements si nécessaire."

        return message


class ActionHandleCostFollowup(Action):
    """
    Handle follow-up questions about the cost estimation with improved conversation persistence
    using Gemini AI to detect user intent and determine flow direction
    """

    def name(self) -> Text:
        return "action_handle_cost_followup"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        # Get the latest message text
        latest_message = tracker.latest_message.get("text", "")

        # Get the cost estimation data
        cost_estimation_str = tracker.get_slot("cost_estimation")
        if not cost_estimation_str:
            dispatcher.utter_message(
                text="Je n'ai pas d'estimation de coût à expliquer. Veuillez d'abord obtenir une estimation."
            )
            return []

        try:
            # Parse the cost estimation
            cost_estimation = json.loads(cost_estimation_str)

            # Get API key for Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("Gemini API key not found in environment variables.")
                dispatcher.utter_message(
                    text="Je ne peux pas traiter votre demande pour le moment. Veuillez réessayer plus tard."
                )
                return []

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

            # First, use Gemini to determine user intent
            intent_prompt = f"""
            Analysez le message suivant d'un utilisateur parlant avec un assistant de coût de projet:

            "{latest_message}"

            TÂCHE: Déterminez l'intention de l'utilisateur. Répondez UNIQUEMENT avec l'une des catégories suivantes:
            1. QUESTION - L'utilisateur pose une question sur l'estimation ou demande plus d'informations
            2. TERMINER - L'utilisateur indique qu'il a terminé, remercie ou veut conclure la conversation
            3. SUGGESTION - L'utilisateur propose des modifications ou demande des ajustements
            4. AUTRE - Tout autre type de message qui n'est pas clairement une question, une fin ou une suggestion

            Répondez seulement avec le code correspondant (QUESTION, TERMINER, SUGGESTION ou AUTRE) sans autre texte.
            """

            # Call the API to determine intent
            intent_response = model.generate_content(intent_prompt)
            user_intent = intent_response.text.strip().upper()

            logger.info(f"Detected user intent: {user_intent}")

            # Prepare the cost estimation data for the response prompt
            cost_data = {
                "total_salary_cost": f"{cost_estimation.get('total_salary_cost', 0):,.2f}".replace(',', ' ').replace(
                    '.', ','),
                "overhead_cost": f"{cost_estimation.get('overhead_cost', 0):,.2f}".replace(',', ' ').replace('.', ','),
                "total_project_cost": f"{cost_estimation.get('total_project_cost', 0):,.2f}".replace(',', ' ').replace(
                    '.', ','),
                "skill_costs": cost_estimation.get("skill_costs", {}),
                "fg_percentage": cost_estimation.get("fg_percentage", 0),
                "project_duration": cost_estimation.get("project_duration", tracker.get_slot("duration"))
            }

            # Initialize slots_to_set list to store individual slot values
            slots_to_set = []
            for key, value in cost_data.items():
                if key != "skill_costs":  # Don't set complex objects as slots
                    slots_to_set.append(SlotSet(key, value))

            # Get recommendations from slot
            recommendations_str = tracker.get_slot("recommendations")
            recommendations = json.loads(recommendations_str) if recommendations_str else {}

            # If user wants to terminate the conversation
            if user_intent == "TERMINER":
                dispatcher.utter_message(
                    text="Parfait ! J'espère que cette estimation de coût vous a été utile. N'hésitez pas à me solliciter si vous avez d'autres besoins."
                )
                # Add the completion slot to the list of slots to set
                return [SlotSet("cost_estimation_complete", True)] + slots_to_set

            # Enhanced prompt for answering the question
            response_prompt = f"""
            Vous êtes un assistant expert en gestion de projet et estimation de coûts.

            Un utilisateur a reçu les recommandations suivantes pour son projet:
            {json.dumps(recommendations, ensure_ascii=False, indent=2)}

            Voici les détails de l'estimation de coût du projet:
            - Durée du projet: {cost_data['project_duration']} mois
            - Coût total des salaires: {cost_data['total_salary_cost']}€
            - Pourcentage de frais généraux (FG%): {cost_data['fg_percentage']}%
            - Frais généraux: {cost_data['overhead_cost']}€
            - Coût total estimé: {cost_data['total_project_cost']}€

            Détail des coûts par compétence:
            {json.dumps(cost_data['skill_costs'], ensure_ascii=False, indent=2)}

            L'utilisateur a envoyé le message suivant:
            "{latest_message}"

            Selon l'analyse de l'intention, l'utilisateur a une intention de type: {user_intent}

            TÂCHE: Répondez de manière pertinente à l'utilisateur en fonction de son message et de son intention.
            - Soyez précis et direct
            - Utilisez un ton professionnel mais accessible
            - Si c'est une QUESTION, donnez une réponse complète et détaillée
            - Si c'est une SUGGESTION, discutez des modifications possibles et de leur impact sur le coût
            - Si c'est AUTRE, essayez de comprendre ce que l'utilisateur veut et orientez la conversation vers les coûts du projet

            Répondez UNIQUEMENT en français.
            """

            # Call the API for the detailed response
            response = model.generate_content(response_prompt)

            # Extract the response text
            generated_text = ""
            if hasattr(response, 'text') and response.text:
                generated_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                generated_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                logger.error("Could not extract text from Gemini API response.")
                dispatcher.utter_message(
                    text="Je ne peux pas traiter votre demande pour le moment. Veuillez réessayer plus tard."
                )
                return slots_to_set  # Still return the slots we've set

            # Add a follow-up prompt that varies based on the context of the conversation
            follow_up_prompt = f"""
            En gardant le contexte de cette conversation, générez une invitation à poursuivre la discussion qui soit:
            - Naturelle et non-répétitive
            - Adaptée à l'intention détectée ({user_intent})
            - Brève (une phrase)
            - En français

            Répondez uniquement avec cette phrase d'invitation, sans autre texte.
            """

            # Get follow-up text
            follow_up_response = model.generate_content(follow_up_prompt)
            follow_up_text = follow_up_response.text.strip()

            # Send the final response
            dispatcher.utter_message(text=generated_text + "\n\n" + follow_up_text)
            logger.info("Response sent successfully")

            # Return the slots we've set
            return slots_to_set

        except Exception as e:
            logger.exception(f"Error handling cost followup: {e}")
            dispatcher.utter_message(
                text="Je suis désolé, je n'ai pas pu traiter votre demande. Pourriez-vous reformuler ou préciser votre question?"
            )
            return []


class ActionPresentCostEstimation(Action):
    """
    Present the cost estimation with properly formatted values from slots
    """

    def name(self) -> Text:
        return "action_present_cost_estimation"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict
    ) -> List[Dict[str, Any]]:
        try:
            # Get the cost estimation data from slots
            cost_estimation_str = tracker.get_slot("cost_estimation")

            if not cost_estimation_str:
                dispatcher.utter_message(
                    text="Je n'ai pas d'estimation de coût à présenter. Veuillez d'abord compléter les étapes précédentes."
                )
                return []

            # Parse the cost estimation JSON
            cost_estimation = json.loads(cost_estimation_str)

            # Format currency values with French notation (space as thousands separator, comma as decimal)
            total_salary_cost = '{:,.2f}'.format(cost_estimation.get("total_salary_cost", 0)).replace(',', ' ').replace(
                '.', ',')
            overhead_cost = '{:,.2f}'.format(cost_estimation.get("overhead_cost", 0)).replace(',', ' ').replace('.',
                                                                                                                ',')
            total_project_cost = '{:,.2f}'.format(cost_estimation.get("total_project_cost", 0)).replace(',',
                                                                                                        ' ').replace(
                '.', ',')

            # Get other necessary values
            project_duration = cost_estimation.get("project_duration", tracker.get_slot("duration"))
            fg_percentage = cost_estimation.get("fg_percentage", 0)

            # Send the formatted message
            dispatcher.utter_message(
                template="utter_present_cost_estimation",
                duration=project_duration,
                total_salary_cost=total_salary_cost,
                fg_percentage=fg_percentage,
                overhead_cost=overhead_cost,
                total_project_cost=total_project_cost
            )

            logger.info("Cost estimation presented successfully")
            return []

        except Exception as e:
            logger.exception(f"Error presenting cost estimation: {e}")
            dispatcher.utter_message(
                text="Je suis désolé, une erreur s'est produite lors de la présentation de l'estimation de coût. Veuillez réessayer."
            )
            return []



#    risk estimation

CSV_FILE = "Risque_total_projets.csv"
RAG_FIELDS = [
    "Nom du projet", "Description", "Secteur", "Pays",
    "Budget Initial (k€)", "Budget Final (k€)", "Écart Budget (%)",
    "Taux Imposition (%)", "Complexité Technique (1-5)", "Mode de Paiement",
    "Qualité des Exigences (1-5)", "Durée Estimée (mois)"
]
EMBED_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash-preview-04-17"  # Or gemini-2.0-pro if you want
API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDINGS_CACHE_FILE = "project_embeddings.pkl"
INDEX_CACHE_FILE = "project_faiss_index.bin"
TEXTS_CACHE_FILE = "project_texts.pkl"

# Configure genai
genai.configure(api_key=API_KEY)

# Global variables to store loaded data
df = None
project_texts = []
index = None


def initialize_rag_system():
    """Initialize the RAG system by loading or creating embeddings and index."""
    global df, project_texts, index

    # Load DataFrame
    df = pd.read_csv(CSV_FILE)

    # Check if cached files exist
    if (os.path.exists(EMBEDDINGS_CACHE_FILE) and
            os.path.exists(INDEX_CACHE_FILE) and
            os.path.exists(TEXTS_CACHE_FILE)):
        logger.info("Loading cached embeddings and index...")
        # Load project texts
        with open(TEXTS_CACHE_FILE, 'rb') as f:
            project_texts = pickle.load(f)

        # Load index
        index = faiss.read_index(INDEX_CACHE_FILE)

        logger.info("Cached data loaded successfully!")
        return

    # If no cached files, compute everything
    logger.info("No cached embeddings found. Computing new embeddings...")

    # Create project texts
    project_texts = []
    for i, row in df.iterrows():
        txt = "\n".join([f"{field}: {row.get(field, '')}" for field in RAG_FIELDS if field in row])
        project_texts.append(txt)

    # Embed all projects (batched)
    embeddings = []
    batch_size = 8
    for i in range(0, len(project_texts), batch_size):
        batch = project_texts[i:i + batch_size]
        try:
            result = genai.embed_content(model=EMBED_MODEL, content=batch, task_type="RETRIEVAL_DOCUMENT")
            logger.info(f"Embedded batch {i // batch_size + 1}/{(len(project_texts) + batch_size - 1) // batch_size}")

            # The Gemini API returns a dict with an 'embedding' key containing a list of embeddings
            batch_emb = [np.array(vec, dtype=np.float32) for vec in result["embedding"]]
            embeddings.extend(batch_emb)
        except Exception as e:
            logger.error(f"Error embedding batch {i // batch_size + 1}: {e}")
            raise

    # Create FAISS index
    embeddings_array = np.vstack(embeddings)
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    # Save to disk for future use
    with open(TEXTS_CACHE_FILE, 'wb') as f:
        pickle.dump(project_texts, f)

    with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
        pickle.dump(embeddings_array, f)

    faiss.write_index(index, INDEX_CACHE_FILE)

    logger.info("Embeddings computed and cached successfully!")


# Initialize the RAG system when the module is imported
initialize_rag_system()


def slot_key(field):
    """Convert CSV field to slot name: e.g., "Nom du projet" -> "nom_du_projet" """
    return field.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").replace("/",
                                                                                                         "_").replace(
        "-", "_").replace("é", "e").replace("à", "a").replace("è", "e").replace("ê", "e").replace("ç", "c")


# Canonical slot names in your domain
RISK_SLOT_TO_DISPLAY = {
    "project_name": "Nom du projet",
    "description": "Description",
    "sector": "Secteur",
    "pays": "Pays",
    "budget_initial_k": "Budget Initial (k€)",
    "budget_final_k": "Budget Final (k€)",
    "ecart_budget_pct": "Écart Budget (%)",
    "taux_imposition_pct": "Taux Imposition (%)",
    "complexity": "Complexité Technique (1-5)",
    "mode_de_paiement": "Mode de Paiement",
    "qualite_des_exigences_1_5": "Qualité des Exigences (1-5)",
    "duration": "Durée Estimée (mois)",
}
RISK_SLOTS = list(RISK_SLOT_TO_DISPLAY.keys())

# Fields that are assumed to be already available
ALREADY_AVAILABLE_FIELDS = ["project_name", "description", "complexity", "sector"]

API_KEY = os.getenv("GEMINI_API_KEY")

# Configure genai
genai.configure(api_key=API_KEY)


class ActionStartRiskEstimation(Action):
    def name(self) -> Text:
        return "action_start_risk_estimation"

    def run(self, dispatcher, tracker, domain) -> List[Dict[Text, Any]]:
        # Get values of already available fields
        already_available_values = {}
        for field in ALREADY_AVAILABLE_FIELDS:
            value = tracker.get_slot(field)
            if value:
                already_available_values[field] = value

        fields_info = ", ".join([f"{RISK_SLOT_TO_DISPLAY[k]}: {v}" for k, v in already_available_values.items()])

        dispatcher.utter_message(
            f"Je vais vous aider à estimer les risques de votre projet. "
            f"J'ai déjà les informations suivantes:\n{fields_info}\n"
            f"Veuillez me fournir les autres détails nécessaires."
        )

        # Reset only the slots that need to be collected
        reset_events = []
        for slot in RISK_SLOTS:
            if slot not in ALREADY_AVAILABLE_FIELDS:
                reset_events.append(SlotSet(slot, None))

        reset_events.append(SlotSet("risk_fields_filled", False))
        reset_events.append(SlotSet("rag_risk_output", None))
        return reset_events


def extract_fields_with_llm(user_message, missing_fields_display):
    """Use Gemini to extract field values from user message"""
    try:
        # Create a prompt for the LLM to extract field values
        prompt = f"""
        Extrayez les informations suivantes du message de l'utilisateur et retournez-les au format JSON.
        Champs à extraire: {', '.join(missing_fields_display)}

        Message de l'utilisateur: "{user_message}"

        Retournez seulement un objet JSON valide avec les champs trouvés. Si une information n'est pas présente dans le message, ne l'incluez pas dans le JSON.
        Par exemple: {{"Pays": "France", "Budget Initial (k€)": 100}}
        """

        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(prompt)

        # Parse the LLM response to extract the JSON
        response_text = response.text
        # Find JSON in the response (in case there's any additional text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            extracted_fields = json.loads(json_str)
            logger.info(f"LLM extracted fields: {extracted_fields}")
            return extracted_fields
        else:
            logger.error("Failed to find valid JSON in LLM response")
            return {}

    except Exception as e:
        logger.error(f"Error during LLM field extraction: {e}")
        return {}


def display_to_slot_name(display_name):
    """Convert display name back to slot name"""
    for slot, display in RISK_SLOT_TO_DISPLAY.items():
        if display == display_name:
            return slot
    return None


class ActionCollectRiskFields(Action):
    def name(self) -> Text:
        return "action_collect_risk_fields"

    def run(self, dispatcher, tracker, domain) -> List[Dict[Text, Any]]:
        # Get the latest user message
        latest_user_message = tracker.latest_message.get("text", "")

        # Check which risk slots are still missing (excluding already available fields)
        missing_slots = [
            slot for slot in RISK_SLOTS
            if slot not in ALREADY_AVAILABLE_FIELDS and not tracker.get_slot(slot)
        ]

        missing_fields_display = [RISK_SLOT_TO_DISPLAY[slot] for slot in missing_slots]

        logger.info(f"Missing risk fields: {missing_fields_display}")

        if not missing_slots:
            # All fields collected
            return [SlotSet("risk_fields_filled", True)]

        # Try to extract values from the user message using LLM
        if latest_user_message and latest_user_message.strip():
            extracted_fields = extract_fields_with_llm(latest_user_message, missing_fields_display)

            # Set slots based on extracted values
            slot_events = []
            for field_display, value in extracted_fields.items():
                slot_name = display_to_slot_name(field_display)
                if slot_name and slot_name in missing_slots:
                    slot_events.append(SlotSet(slot_name, value))
                    missing_slots.remove(slot_name)
                    missing_fields_display.remove(field_display)

            # Provide feedback on what was understood
            if slot_events:
                understood_fields = ", ".join([
                    f"{RISK_SLOT_TO_DISPLAY.get(event['name'], event['name'])}: {event['value']}"
                    for event in slot_events
                ])
                dispatcher.utter_message(f"J'ai compris les informations suivantes:\n{understood_fields}")

            # If all fields are now collected
            if not missing_slots:
                dispatcher.utter_message(
                    "J'ai maintenant toutes les informations nécessaires pour l'estimation des risques.")
                slot_events.append(SlotSet("risk_fields_filled", True))
                return slot_events

            # If we extracted some but not all fields
            if slot_events:
                # Continue asking for remaining fields
                if len(missing_fields_display) == 1:
                    msg = f"Merci de me fournir également : {missing_fields_display[0]}"
                else:
                    msg = "Merci de me fournir également : " + ", ".join(missing_fields_display)
                dispatcher.utter_message(msg)
                return slot_events

        # If no fields were extracted or this is the first request
        logger.info("Collecting risk fields...")
        if len(missing_fields_display) == 1:
            msg = f"Merci de me fournir la valeur suivante pour l'estimation des risques : {missing_fields_display[0]}"
        else:
            msg = "Merci de me fournir les valeurs suivantes pour l'estimation des risques : " + ", ".join(
                missing_fields_display)
        dispatcher.utter_message(msg)
        return [SlotSet("risk_fields_filled", False)]


def map_domain_slots_to_csv_fields(tracker):
    """Map the domain slots to CSV field names used in the RAG system."""

    # This mapping connects your existing domain slots to the CSV field names
    slot_to_csv_mapping = {
        "project_name": "Nom du projet",
        "description": "Description",
        "sector": "Secteur",
        "pays": "Pays",
        "budget_initial_k": "Budget Initial (k€)",
        "budget_final_k": "Budget Final (k€)",
        "ecart_budget_pct": "Écart Budget (%)",
        "taux_imposition_pct": "Taux Imposition (%)",
        "complexity": "Complexité Technique (1-5)",
        "mode_de_paiement": "Mode de Paiement",
        "qualite_des_exigences_1_5": "Qualité des Exigences (1-5)",
        "duration": "Durée Estimée (mois)"
    }

    # Create the input dictionary with CSV field names
    input_dict = {}
    for slot_name, csv_field in slot_to_csv_mapping.items():
        value = tracker.get_slot(slot_name)
        if value is not None and str(value).strip() != "":
            input_dict[csv_field] = value

    logger.info(f"Input dictionary for RAG system: {input_dict}")

    return input_dict


class ActionPresentRagRiskAssessment(Action):
    def name(self) -> Text:
        return "action_present_rag_risk_assessment"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logger.info("Starting risk assessment process")

        # 1. Map slots to fields and create query
        try:
            input_dict = map_domain_slots_to_csv_fields(tracker)
            logger.info(f"Mapped slots to fields: {json.dumps(input_dict, ensure_ascii=False)}")

            # Filter out None values and create query text
            filtered_dict = {k: v for k, v in input_dict.items() if v is not None}
            logger.info(f"Using {len(filtered_dict)}/{len(input_dict)} fields with non-None values")
            query_text = "\n".join([f"{k}: {v}" for k, v in filtered_dict.items()])
            logger.debug(f"Constructed query text: {query_text}")

            if not filtered_dict:
                logger.warning("No valid input data found in slots")
                dispatcher.utter_message("Impossible d'évaluer les risques sans données d'entrée valides.")
                return []
        except Exception as e:
            logger.error(f"Error mapping slots to fields: {str(e)}")
            logger.error(traceback.format_exc())
            dispatcher.utter_message("Une erreur s'est produite lors de la préparation des données d'entrée.")
            return []

        try:
            # 2. Embed query
            logger.info("Embedding query with model: " + EMBED_MODEL)
            start_time = time.time()
            result = genai.embed_content(model=EMBED_MODEL, content=[query_text], task_type="RETRIEVAL_QUERY")
            embedding_time = time.time() - start_time
            logger.info(f"Embedding completed in {embedding_time:.2f}s")

            # Adapte le nom de la clé ici selon ta vraie sortie
            if not result or "embedding" not in result or not result["embedding"]:
                logger.error(f"Invalid embedding response: {result}")
                dispatcher.utter_message("Erreur lors de la génération des embeddings.")
                return []

            query_emb = np.array(result["embedding"][0], dtype=np.float32).reshape(1, -1)
            logger.info(f"Embedding shape: {query_emb.shape}")
            logger.debug(f"Embedding values (first 5): {query_emb[0][:5]}")

            # 3. Search in index for most similar projects
            k = 4  # Number of similar projects to retrieve
            logger.info(f"Searching index for {k} most similar projects")

            # Log index status
            logger.info(f"Index status - Size: {index.ntotal}, Dimension: {index.d}")

            start_time = time.time()
            distances, indices = index.search(query_emb, k)
            search_time = time.time() - start_time
            logger.info(f"Vector search completed in {search_time:.2f}s")

            # Check if we have valid results
            if len(indices[0]) == 0:
                logger.warning("No similar projects found in index")
                dispatcher.utter_message("Aucun projet similaire n'a été trouvé pour l'évaluation des risques.")
                return []

            logger.info(f"Found {len(indices[0])} similar projects with indices: {indices[0]}")
            logger.info(f"Similarity distances: {distances[0]}")

            similar_projects = []
            for i, idx in enumerate(indices[0]):
                logger.debug(f"Similar project {i + 1} (idx={idx}, distance={distances[0][i]:.4f})")
                if idx < len(project_texts):
                    similar_projects.append(project_texts[idx])
                else:
                    logger.error(f"Index {idx} out of bounds for project_texts (length: {len(project_texts)})")

            if not similar_projects:
                logger.warning("Failed to retrieve any similar project texts")
                dispatcher.utter_message("Impossible de récupérer des projets similaires pour l'évaluation.")
                return []

            context_block = "\n---\n".join(similar_projects)
            logger.debug(
                f"Context block constructed with {len(similar_projects)} projects, total length: {len(context_block)}")

            # 4. Build Gemini prompt for risk estimation
            logger.info("Building prompt for risk assessment")
            prompt = (
                f"Voici les informations d'un projet pour lequel on souhaite estimer les risques :\n"
                f"{query_text}\n\n"
                f"Voici {k} projets similaires :\n"
                f"{context_block}\n\n"
                f"Tâche : En te basant sur ces projets similaires, donne une estimation des risques pour le projet en question, sous forme de JSON :\n"
                f'{{"risque_delais": ..., "risque_financement": ..., "risque_penalites": ..., "risque_fiscalite": ..., "risque_technique": ..., "risque_frais": ..., "risque_moyen": ...}} '
                f"Où chaque champ est un pourcentage sur 20 (ex: 12.7, représentant 12.7%), et 'risque_moyen' est la moyenne des 6 risques. "
                f"**ATTENTION : chaque valeur de risque doit être comprise entre 0 (faible) et 20 (très élevée).** "
                f"Ajoute aussi un court commentaire général. Ne fais AUCUNE autre sortie que le JSON demandé et le commentaire (séparé)."
                )
            logger.debug(f"Prompt length: {len(prompt)} characters")

            # 5. Generate risk assessment
            logger.info(f"Generating risk assessment with model: {LLM_MODEL}")
            start_time = time.time()
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(
                [{"role": "user", "parts": [{"text": prompt}]}]
            )
            generation_time = time.time() - start_time
            logger.info(f"Risk assessment generation completed in {generation_time:.2f}s")

            # Extract and validate response
            risk_output = response.text.strip()
            logger.debug(f"Raw LLM response: {risk_output}")

            if not risk_output:
                logger.warning("Empty response from LLM")
                dispatcher.utter_message("Le modèle n'a pas généré d'évaluation des risques.")
                return []

            # Basic validation of JSON output (optional)
            try:
                # Try to find JSON in the output
                json_start = risk_output.find('{')
                json_end = risk_output.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = risk_output[json_start:json_end]
                    risk_json = json.loads(json_str)
                    logger.info(f"Validated risk JSON: {risk_json}")
                    # Check if all expected fields are present
                    expected_fields = ["risque_delais", "risque_financement", "risque_penalites",
                                       "risque_fiscalite", "risque_technique", "risque_frais", "risque_moyen"]
                    missing_fields = [f for f in expected_fields if f not in risk_json]
                    if missing_fields:
                        logger.warning(f"Missing fields in risk JSON: {missing_fields}")
            except json.JSONDecodeError as e:
                logger.warning(f"Could not validate JSON in output: {e}")
                # Continue anyway as we'll return the raw text

            # 6. Present to user
            logger.info("Presenting risk assessment to user")
            dispatcher.utter_message("Voici l'estimation des risques générée :")
            dispatcher.utter_message(risk_output)

            logger.info("Risk assessment process completed successfully")
            return [SlotSet("rag_risk_output", risk_output)]

        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            logger.error(traceback.format_exc())
            dispatcher.utter_message("Une erreur s'est produite lors de l'estimation des risques. Veuillez réessayer.")
            return []


class ActionHandleRagRiskFollowup(Action):
    def name(self):
        return "action_handle_rag_risk_followup"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get("text", "")
        risk_output = tracker.get_slot("rag_risk_output")

        if not risk_output:
            dispatcher.utter_message("Je n'ai pas d'estimation de risque à expliquer.")
            return []

        # Compose followup prompt
        prompt = (
            f"Voici l'estimation des risques pour un projet (générée précédemment) :\n"
            f"{risk_output}\n\n"
            f"Un utilisateur pose la question ou demande une recommandation suivante :\n"
            f"\"{user_msg}\"\n"
            f"Réponds de façon professionnelle, claire et concise, en français."
        )

        try:
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(
                [{"role": "user", "parts": [{"text": prompt}]}]
            )
            dispatcher.utter_message(response.text.strip())
        except Exception as e:
            logger.error(f"Error handling followup: {e}")
            dispatcher.utter_message("Désolé, je n'ai pas pu traiter votre demande. Pourriez-vous reformuler?")

        return []
