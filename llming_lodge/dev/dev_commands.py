"""Built-in /dev command dispatcher for llming-lodge.

Handles all ``/dev ...`` commands natively within llming-lodge:
- ``/dev enable``       — enable dev mode (raw data inspection, show session ID)
- ``/dev disable``      — disable dev mode
- ``/dev remote``       — activate remote mode (MongoDB task polling)
- ``/dev remote stop``  — deactivate remote mode
- ``/dev enable mcp:*`` — activate registered dev MCPs
- ``/dev budget <email>``         — show budget for user
- ``/dev budget <email> set <n>`` — set user's monthly budget to n EUR

Returns the response string if the command was handled, or None if the
message should pass through to the LLM (or an app-level intercept).
"""

import logging
from typing import Optional

from llming_lodge.dev.dev_mcp import try_activate_dev_mcp
from llming_lodge.dev.remote_mode import activate_remote_mode, deactivate_remote_mode

logger = logging.getLogger(__name__)


async def _handle_budget_command(args: str, controller) -> str:
    """Handle /dev budget <email> [set <amount>]."""
    parts = args.strip().split()
    if not parts:
        return (
            "**Usage**\n\n"
            "`/dev budget user@example.com` — show budget\n\n"
            "`/dev budget user@example.com set 100` — set monthly budget to 100 EUR"
        )

    email = parts[0].lower()
    if "@" not in email:
        return f"Invalid email: `{email}`"

    # Build budget limits for the target user
    budget_handler = getattr(controller, "_budget_limits_for_user", None)
    if not budget_handler:
        return "Budget management not available (no `_budget_limits_for_user` handler registered)."

    limits = budget_handler(email)
    if not limits:
        return f"No budget limits configured for `{email}`."

    # SET subcommand
    if len(parts) >= 3 and parts[1] == "set":
        try:
            new_amount = float(parts[2])
        except ValueError:
            return f"Invalid amount: `{parts[2]}`"
        if new_amount < 0:
            return "Amount must be non-negative."

        from llming_models.budget.mongodb_budget_limit import MongoDBBudgetLimit
        user_limit = next((l for l in limits if isinstance(l, MongoDBBudgetLimit) and l.name.startswith("users:")), None)
        if not user_limit:
            return f"No per-user MongoDB budget limit found for `{email}`."

        await user_limit.set_amount_async(new_amount)
        return f"Budget for `{email}` set to **{new_amount:.2f} EUR/month**."

    # QUERY — show current budget
    lines = [f"**Budget for `{email}`**\n"]
    from llming_models.budget.mongodb_budget_limit import MongoDBBudgetLimit
    for limit in limits:
        if isinstance(limit, MongoDBBudgetLimit):
            effective_amount = await limit._get_effective_amount()
            used = await limit.get_usage_async()
            available = max(effective_amount - used, 0.0)
            is_override = effective_amount != limit._default_amount
            amount_str = f"{effective_amount:.2f}"
            if is_override:
                amount_str += f" (default: {limit._default_amount:.2f})"
            lines.append(
                f"| `{limit.name}` | {limit.period.value} | "
                f"Limit: {amount_str} EUR | Used: {used:.4f} EUR | "
                f"Available: {available:.4f} EUR |"
            )
        else:
            available = await limit.get_available_budget_async()
            lines.append(f"| `{limit.name}` | {limit.period.value} | Available: {available:.4f} EUR |")

    return "\n".join(lines)


async def handle_dev_command(text: str, controller) -> Optional[str]:
    """Dispatch a /dev command. Returns response text or None if not a dev command."""
    normalized = text.strip().lower()

    if not normalized.startswith("/dev"):
        return None

    logger.info("[DEV] Command received: %r (is_dev=%s)", normalized, getattr(controller, "_is_dev", False))

    # Requires dev_tools permission
    if not getattr(controller, "_is_dev", False):
        return None

    if normalized == "/dev enable":
        sid = controller.session_id or "unknown"
        await controller._send({"type": "dev_mode", "enabled": True})
        return (
            f"**Dev mode enabled**\n\n"
            f"Conversation ID: `{sid}`\n\n"
            f"*Cmd+Shift+Click (Mac) or Ctrl+Shift+Click on any document "
            f"or visualization to view raw data.*"
        )

    if normalized == "/dev disable":
        await controller._send({"type": "dev_mode", "enabled": False})
        return "**Dev mode disabled**"

    if normalized == "/dev remote":
        return await activate_remote_mode(controller)

    if normalized == "/dev remote stop":
        return deactivate_remote_mode(controller)

    if normalized == "/dev test":
        return (
            "**Test Protocol**\n\n"
            "Open [Test Protocol](/test-protocol) to begin."
        )

    # /dev budget <email> [set <amount>]
    if normalized.startswith("/dev budget"):
        args = text.strip()[len("/dev budget"):].strip()
        return await _handle_budget_command(args, controller)

    # Try dev MCP activation (e.g. "/dev enable mcp:sapperlot")
    return await try_activate_dev_mcp(text, controller)
