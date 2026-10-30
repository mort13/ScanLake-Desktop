"""Bring the local profile and the Discord account together (port of Capture's reconcile()).

1. Link the local ScanLake ID to the account, unless it is an unused fresh one (the account
   already has a primary ID and nothing was captured under the local one yet).
2. The account's primary ID wins on this device, so all devices upload under one ID.
3. Empty local name/org are filled from the account; if the account's copy is empty, the local
   values are pushed up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from scanlake_desktop.auth.client import AuthClient, AuthError, Me, UserIdConflict, UserIdLimit

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalProfile:
    user_id: str
    user_name: str
    org: str


@dataclass(frozen=True)
class Reconciled:
    me: Me
    profile: LocalProfile
    notice: str | None = None

    def changed(self, before: LocalProfile) -> bool:
        return self.profile != before


def reconcile(
    client: AuthClient, token: str, me: Me, local: LocalProfile, has_local_data: bool
) -> Reconciled:
    notice: str | None = None
    if local.user_id not in me.user_ids and (me.primary_user_id is None or has_local_data):
        try:
            me = client.link_user_id(token, local.user_id)
        except (UserIdConflict, UserIdLimit) as err:
            notice = str(err)
        except AuthError as err:
            log.warning("linking the local ID failed: %s", err)

    profile = LocalProfile(
        user_id=me.primary_user_id or local.user_id,
        user_name=local.user_name or me.user_name,
        org=local.org or me.org,
    )
    if (not me.user_name and profile.user_name) or (not me.org and profile.org):
        try:
            me = client.put_profile(token, profile.user_name, profile.org)
        except AuthError as err:
            log.warning("profile sync failed: %s", err)
    return Reconciled(me=me, profile=profile, notice=notice)
