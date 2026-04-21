from __future__ import annotations

from secknow.vector_store.models import BaselineBundle
from secknow.vector_store.services.vector_service import VectorInfrastructureService


class BaselineRetriever:
    """Thin 4.4 adapter over 4.3 baseline access."""

    def __init__(self, vector_service: VectorInfrastructureService) -> None:
        self.vector_service = vector_service

    def get_baseline(self, zone_id: str) -> BaselineBundle:
        return self.vector_service.get_baseline(zone_id=zone_id)
