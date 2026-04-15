from app.nlp.intent.classifier import classifier
from app.nlp.ner.extractor import NERExtractor
extractor = NERExtractor()
from app.models.schemas import QueryResponse, IntentResult, EntityResult
from app.services.handlers import directions_handler, hours_handler, proximity_handler, events_handler

INTENT_HANDLERS = {
    "DIRECTIONS": directions_handler.handle,
    "FACILITY_HOURS": hours_handler.handle,
    "PROXIMITY": proximity_handler.handle,
    "EVENTS": events_handler.handle,
}

async def handle_query(query: str) -> QueryResponse:
    raw_entities = extractor.extract_flat(query)
    entities = []
    for text, label in raw_entities:
        start = query.find(text)
        end = start + len(text) if start != -1 else 0
        entities.append(EntityResult(text=text, label=label, start=start, end=end))

    try:
        intent_label, confidence = classifier.predict(query)
    except RuntimeError:
        return QueryResponse(
            query=query,
            intent=IntentResult(intent="UNKNOWN", confidence=0.0),
            entities=entities,
            response="I'm still being trained. Please check back soon.",
            source="fallback"
        )

    intent = IntentResult(intent=intent_label, confidence=confidence)
    handler = INTENT_HANDLERS.get(intent_label)

    if handler:
        response_text, data, source = await handler(query, entities)
    else:
        response_text = "I'm not sure how to help with that. Try asking about directions, opening hours, nearby facilities, or campus events."
        data = None
        source = "fallback"

    return QueryResponse(query=query, intent=intent, entities=entities, response=response_text, data=data, source=source)