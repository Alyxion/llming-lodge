"""Abstract directory / people-search service.

Used by the email composer's contact search, org chart widgets,
Teams people picker, etc. — anything that needs to find users.
"""

from abc import ABC, abstractmethod


class DirectoryService(ABC):

    @abstractmethod
    async def search_people(self, query: str, limit: int = 10) -> list[dict]:
        """Search for people matching *query*.

        Returns a list of dicts with keys:
        ``name``, ``email``, ``title``, ``department``, ``photo_url``.
        """
