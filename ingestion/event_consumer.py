"""Kafka event consumer for the recommendation platform."""

import json
import logging
import asyncio
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConsumerConfig:
    bootstrap_servers: str
    group_id: str
    topics: List[str]
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    max_poll_records: int = 500


class EventConsumer:
    """
    Async Kafka consumer that routes events to registered handlers.
    Supports graceful shutdown, dead-letter queuing, and metrics.
    """

    def __init__(self, config: ConsumerConfig):
        self.config = config
        self._handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._consumer = None
        self._dlq: List[Dict[str, Any]] = []
        self._processed_count = 0
        self._error_count = 0

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler function for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")

    async def start(self) -> None:
        """Start consuming events from Kafka."""
        logger.info(f"Starting EventConsumer for topics: {self.config.topics}")
        self._running = True
        try:
            # In production, replace with aiokafka.AIOKafkaConsumer
            await self._consume_loop()
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            raise
        finally:
            await self.stop()

    async def _consume_loop(self) -> None:
        """Main consumption loop."""
        while self._running:
            try:
                # Simulate message batch fetch
                messages = await self._fetch_messages()
                await asyncio.gather(*[
                    self._process_message(msg) for msg in messages
                ])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                self._error_count += 1
                await asyncio.sleep(1)

    async def _fetch_messages(self) -> List[Dict[str, Any]]:
        """Fetch a batch of messages. Override with real Kafka implementation."""
        await asyncio.sleep(0.1)
        return []

    async def _process_message(self, message: Dict[str, Any]) -> None:
        """Process a single message through registered handlers."""
        try:
            event_type = message.get("event_type", "unknown")
            handlers = self._handlers.get(event_type, []) + self._handlers.get("*", [])
            await asyncio.gather(*[handler(message) for handler in handlers])
            self._processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process message: {e}, message: {message}")
            self._dlq.append({"message": message, "error": str(e)})
            self._error_count += 1

    async def stop(self) -> None:
        """Gracefully stop the consumer."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
        logger.info(
            f"Consumer stopped. Processed: {self._processed_count}, "
            f"Errors: {self._error_count}, DLQ: {len(self._dlq)}"
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "processed": self._processed_count,
            "errors": self._error_count,
            "dlq_size": len(self._dlq),
            "running": self._running,
        }
