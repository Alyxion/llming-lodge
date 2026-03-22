# Bolts — 200 Fast Commands for Everyday Work

Bolts are instant commands typed directly into the chat input. They execute
immediately without an AI round-trip — lookups, calculations, navigation,
and mini-apps at the speed of typing.

## Bolt Types

| Type | Behavior |
|------|----------|
| **Lookup** | Instant data retrieval, result shown inline in chat |
| **Open** | Opens a URL, portal, or external page in a new tab |
| **App** | Opens an embedded mini-app (calculator, timer, converter…) rendered as a doc plugin |
| **Action** | Triggers a side effect (create ticket, send email, book room…) |
| **Prompt** | Pre-fills a smart AI prompt for the user to refine before sending |

## Design Principles

- Bolts **live inside nudges** (droplets). Each nudge can define bolts in its
  **Bolts tab**. This ties bolt availability, permissions, and MCP tool access
  to the nudge that owns them.
- The UI label "Bolts" is **host-app configurable** (e.g. Lechler shows "Jets").
- Both **English** and **alias** command names are recognized (i18n-ready,
  extensible to more languages).
- **Two trigger modes**: explicit `/command` with autocomplete, or **passive
  regex auto-detection** that surfaces floating suggestion chips above the input.
- **Shift+Enter** or **click** on a suggestion chip executes the bolt.
  Normal Enter sends to AI as usual.
- Three **action types** (extensible): URL redirect, prompt template, MCP tool call.
- Bolts that return data render results as **inline cards** or **doc plugin blocks**
  (reusing the existing document/rich_mcp infrastructure).

---

## 1. People & Organization (1-20)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 1 | `/who <name>` | `/wer <name>` | Lookup | Look up a colleague -- phone, email, department, office, photo |
| 2 | `/org <name>` | `/orgchart <name>` | Lookup | Show org chart around a person (3 levels) |
| 3 | `/team <name>` | `/team <name>` | Lookup | List all members of a team or department |
| 4 | `/manager <name>` | `/chef <name>` | Lookup | Who does this person report to? |
| 5 | `/phone <name>` | `/telefon <name>` | Lookup | Quick phone number lookup (mobile + landline) |
| 6 | `/email <name>` | `/mail <name>` | Lookup | Get someone's email address |
| 7 | `/birthday` | `/geburtstag` | Lookup | Upcoming birthdays this week/month |
| 8 | `/jubilee` | `/jubilaeum` | Lookup | Upcoming work anniversaries |
| 9 | `/newstarters` | `/neue` | Lookup | New employees this month |
| 10 | `/leavers` | `/abgaenge` | Lookup | Departures this month |
| 11 | `/deputy <name>` | `/vertreter <name>` | Lookup | Who covers during absence? |
| 12 | `/dl <department>` | `/vl <abteilung>` | Lookup | Distribution list / email group for a dept |
| 13 | `/site <location>` | `/standort <ort>` | Lookup | Contact info for a company site |
| 14 | `/sites` | `/standorte` | Lookup | List all company locations worldwide |
| 15 | `/ceo` | `/vorstand` | Lookup | Current executive board / management |
| 16 | `/emergency` | `/notfall` | Lookup | Emergency contacts -- first aid, fire, facility, IT hotline |
| 17 | `/available <name>` | `/frei <name>` | Lookup | Is this person free right now? (calendar check) |
| 18 | `/absent` | `/abwesend` | Lookup | Who in my team is out today? |
| 19 | `/headcount <dept>` | `/koepfe <abt>` | Lookup | Headcount of a department |
| 20 | `/intern <name>` | `/praktikant <name>` | Lookup | Current interns & working students |

## 2. Calendar & Scheduling (21-40)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 21 | `/today` | `/heute` | Lookup | My meetings and tasks for today |
| 22 | `/tomorrow` | `/morgen` | Lookup | My schedule for tomorrow |
| 23 | `/week` | `/woche` | Lookup | My calendar this week at a glance |
| 24 | `/next` | `/naechstes` | Lookup | Next meeting -- what, when, where, with whom |
| 25 | `/free <date>` | `/frei <datum>` | Lookup | My free time slots on a given date |
| 26 | `/free2 <name> <date>` | `/frei2 <name> <datum>` | Lookup | Find overlapping free slots with a colleague |
| 27 | `/room <date> <time>` | `/raum <datum> <zeit>` | Lookup | Find available meeting rooms |
| 28 | `/book <room> <time>` | `/buchen <raum> <zeit>` | Action | Quick-book a meeting room |
| 29 | `/invite <names> <time>` | `/einladen <namen> <zeit>` | Action | Send a meeting invite |
| 30 | `/cancel <meeting>` | `/absagen <termin>` | Action | Cancel or decline a meeting |
| 31 | `/reschedule <meeting>` | `/verschieben <termin>` | Prompt | Suggest alternative times and reschedule |
| 32 | `/holiday` | `/feiertag` | Lookup | Next public holidays (by site/country) |
| 33 | `/bridge` | `/brueckentag` | Lookup | Upcoming bridge days worth taking off |
| 34 | `/pto` | `/urlaub` | Lookup | My remaining vacation days this year |
| 35 | `/pto-plan` | `/urlaubsplan` | Lookup | Team vacation overview (who's out when) |
| 36 | `/ooo <dates>` | `/abwesenheit <daten>` | Action | Set out-of-office auto-reply |
| 37 | `/deadline` | `/frist` | Lookup | My upcoming deadlines (from tasks/calendar) |
| 38 | `/agenda <meeting>` | `/agenda <termin>` | Lookup | Show agenda / attachments of a meeting |
| 39 | `/timezone <city>` | `/zeitzone <stadt>` | Lookup | Current time in another timezone |
| 40 | `/clock` | `/uhr` | App | World clock mini-app with company site times |

## 3. Email & Communication (41-58)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 41 | `/mail <name> <subject>` | `/schreiben <name> <betreff>` | Prompt | Quick-draft an email |
| 42 | `/reply` | `/antworten` | Prompt | Draft a reply to the last unread email |
| 43 | `/forward <name>` | `/weiterleiten <name>` | Action | Forward last email to someone |
| 44 | `/unread` | `/ungelesen` | Lookup | Unread email count & top senders |
| 45 | `/followup` | `/nachfassen` | Lookup | Emails flagged for follow-up |
| 46 | `/mailto <team>` | `/mailgruppe <team>` | Prompt | Compose email to a whole team |
| 47 | `/translate <lang>` | `/uebersetzen <sprache>` | Prompt | Translate last message or selected text |
| 48 | `/formal` | `/formell` | Prompt | Rewrite text in formal business tone |
| 49 | `/casual` | `/locker` | Prompt | Rewrite text in friendly casual tone |
| 50 | `/shorten` | `/kuerzen` | Prompt | Make text more concise |
| 51 | `/proofread` | `/korrektur` | Prompt | Check grammar and spelling |
| 52 | `/summarize` | `/zusammenfassen` | Prompt | Summarize the current conversation |
| 53 | `/bullet` | `/stichpunkte` | Prompt | Convert last AI answer to bullet points |
| 54 | `/english` | `/englisch` | Prompt | Translate last message to English |
| 55 | `/german` | `/deutsch` | Prompt | Translate last message to German |
| 56 | `/french` | `/franzoesisch` | Prompt | Translate last message to French |
| 57 | `/chinese` | `/chinesisch` | Prompt | Translate last message to Chinese |
| 58 | `/template <type>` | `/vorlage <typ>` | Prompt | Email template (offer, complaint response, meeting notes...) |

## 4. Canteen & Facilities (59-72)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 59 | `/lunch` | `/mittagessen` | Lookup | Today's canteen menu |
| 60 | `/lunch tomorrow` | `/mittagessen morgen` | Lookup | Tomorrow's canteen menu |
| 61 | `/lunch week` | `/mittagessen woche` | Lookup | Full week canteen overview |
| 62 | `/vegan` | `/vegan` | Lookup | Vegan options this week |
| 63 | `/vegetarian` | `/vegetarisch` | Lookup | Vegetarian options this week |
| 64 | `/coffee` | `/kaffee` | Lookup | Nearest coffee machine / kitchen location |
| 65 | `/parking` | `/parken` | Lookup | Parking info, guest parking, EV charging |
| 66 | `/shuttle` | `/shuttle` | Lookup | Shuttle bus schedule between sites/buildings |
| 67 | `/reception` | `/empfang` | Lookup | Reception phone number & visitor registration |
| 68 | `/visitor <name>` | `/besucher <name>` | Action | Register an expected visitor |
| 69 | `/defect <location>` | `/defekt <ort>` | Action | Report a facility defect (broken light, heating...) |
| 70 | `/cleaning` | `/reinigung` | Lookup | Cleaning schedule / report cleaning issue |
| 71 | `/map` | `/lageplan` | Open | Building map / floor plan |
| 72 | `/locker` | `/schliessfach` | Lookup | Locker assignment info |

## 5. Products & Domain-Specific (73-105)

These are examples for a manufacturing/engineering company. Host apps replace
this section with their own domain (finance, healthcare, retail, etc.).

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 73 | `/product <number>` | `/produkt <nummer>` | Lookup | Look up a product by number |
| 74 | `/search <query>` | `/suche <text>` | Lookup | Search products by description or keyword |
| 75 | `/customer <name>` | `/kunde <name>` | Lookup | Search customer database |
| 76 | `/debtor <id>` | `/debitor <id>` | Lookup | Customer details by debtor number |
| 77 | `/price <product>` | `/preis <produkt>` | Lookup | List price / price group |
| 78 | `/stock <product>` | `/bestand <produkt>` | Lookup | Stock level / warehouse availability |
| 79 | `/leadtime <product>` | `/lieferzeit <produkt>` | Lookup | Expected production/delivery lead time |
| 80 | `/order <number>` | `/auftrag <nummer>` | Lookup | Order status by order number |
| 81 | `/orders <customer>` | `/auftraege <kunde>` | Lookup | Recent orders for a customer |
| 82 | `/offer <number>` | `/angebot <nummer>` | Lookup | Quotation status / details |
| 83 | `/bom <product>` | `/stueckliste <produkt>` | Lookup | Bill of materials |
| 84 | `/drawing <product>` | `/zeichnung <produkt>` | Open | Open technical drawing (PDF) |
| 85 | `/datasheet <product>` | `/datenblatt <produkt>` | Open | Open product data sheet |
| 86 | `/catalog` | `/katalog` | Open | Open the online product catalog |
| 87 | `/compare <p1> <p2>` | `/vergleich <p1> <p2>` | Lookup | Side-by-side comparison of two product specs |
| 88 | `/material <code>` | `/werkstoff <code>` | Lookup | Material code lookup (1.4404, PVDF, PEEK...) |
| 89 | `/competitor <part>` | `/wettbewerber <teil>` | Lookup | Cross-reference competitor part number |
| 90 | `/replace <product>` | `/ersatz <produkt>` | Lookup | Replacement / successor product |
| 91 | `/new-products` | `/neuheiten` | Lookup | Recently launched products |
| 92 | `/discontinued` | `/auslauf` | Lookup | Discontinued products & their replacements |
| 93 | `/certifications <product>` | `/zertifikate <produkt>` | Lookup | Product certifications (FDA, ATEX, CE...) |
| 94 | `/application <industry>` | `/anwendung <branche>` | Lookup | Products recommended for an industry/application |
| 95 | `/cad <product>` | `/cad <produkt>` | Open | Download 3D CAD model (STEP/IGES) |
| 96 | `/quotation <customer>` | `/offerte <kunde>` | Prompt | Start drafting a quotation |
| 97 | `/complaint <order>` | `/reklamation <auftrag>` | Action | Start a complaint / return process |
| 98 | `/sample <product> <customer>` | `/muster <produkt> <kunde>` | Action | Request a product sample |
| 99 | `/visit <customer>` | `/besuch <kunde>` | Prompt | Prepare a customer visit briefing |
| 100 | `/history <customer>` | `/historie <kunde>` | Lookup | Order & interaction history with a customer |
| 101 | `/territory` | `/gebiet` | Lookup | My sales territory / assigned regions |
| 102 | `/domain-calc-1` | `/fach-rechner-1` | Lookup | Domain-specific calculation 1 (e.g. flow rate) |
| 103 | `/domain-calc-2` | `/fach-rechner-2` | Lookup | Domain-specific calculation 2 (e.g. pressure) |
| 104 | `/domain-calc-3` | `/fach-rechner-3` | Lookup | Domain-specific calculation 3 (e.g. coverage) |
| 105 | `/domain-lookup` | `/fach-suche` | Lookup | Domain-specific reference lookup |

## 6. Calculators & Converters (106-125)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 106 | `/calc` | `/rechner` | App | Scientific calculator mini-app |
| 107 | `/convert` | `/umrechner` | App | Unit converter mini-app (pressure, flow, length, temp...) |
| 108 | `/bar2psi <value>` | `/bar2psi <wert>` | Lookup | Quick pressure conversion |
| 109 | `/psi2bar <value>` | `/psi2bar <wert>` | Lookup | Quick pressure conversion |
| 110 | `/lpm2gpm <value>` | `/lpm2gpm <wert>` | Lookup | Flow rate conversion l/min to gpm |
| 111 | `/mm2inch <value>` | `/mm2zoll <wert>` | Lookup | Length conversion mm to inch |
| 112 | `/c2f <value>` | `/c2f <wert>` | Lookup | Temperature conversion C to F |
| 113 | `/f2c <value>` | `/f2c <wert>` | Lookup | Temperature conversion F to C |
| 114 | `/kg2lb <value>` | `/kg2lb <wert>` | Lookup | Weight conversion kg to lbs |
| 115 | `/fx <amount> <from> <to>` | `/fx <betrag> <von> <nach>` | Lookup | Currency conversion (live rates) |
| 116 | `/plot <function>` | `/plot <funktion>` | Prompt | Quick function plot (activates Math nudge) |
| 117 | `/solve <equation>` | `/loesen <gleichung>` | Prompt | Solve equation step by step |
| 118 | `/matrix` | `/matrix` | App | Matrix calculator mini-app |
| 119 | `/statistics` | `/statistik` | App | Statistics calculator (mean, std dev, distribution...) |
| 120 | `/tolerance <nom> <tol>` | `/toleranz <nom> <tol>` | Lookup | Tolerance stack-up calculation |
| 121 | `/thread <size>` | `/gewinde <groesse>` | Lookup | Thread dimensions lookup (M, G, NPT, BSP) |
| 122 | `/hardness <value> <from>` | `/haerte <wert> <von>` | Lookup | Hardness conversion (HRC, HV, HB) |
| 123 | `/density <material>` | `/dichte <werkstoff>` | Lookup | Material density lookup |
| 124 | `/reynolds <v> <d> <fluid>` | `/reynolds <v> <d> <fluid>` | Lookup | Reynolds number calculation |
| 125 | `/viscosity <fluid> <temp>` | `/viskositaet <fluid> <temp>` | Lookup | Fluid viscosity at temperature |

## 7. IT & Systems (126-145)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 126 | `/ticket <description>` | `/ticket <beschreibung>` | Action | Create an IT support ticket |
| 127 | `/tickets` | `/tickets` | Lookup | My open IT tickets |
| 128 | `/vpn` | `/vpn` | Lookup | VPN setup instructions |
| 129 | `/wifi` | `/wlan` | Lookup | WiFi name & password (guest/internal) |
| 130 | `/printer <location>` | `/drucker <ort>` | Lookup | Find nearest printer & setup guide |
| 131 | `/password` | `/passwort` | Open | Password reset portal link |
| 132 | `/2fa` | `/2fa` | Lookup | Two-factor authentication setup guide |
| 133 | `/software <name>` | `/software <name>` | Lookup | How to request or install software |
| 134 | `/status` | `/systemstatus` | Lookup | IT system status (ERP, email, VPN, network) |
| 135 | `/share <path>` | `/laufwerk <pfad>` | Lookup | Find a network share or SharePoint site |
| 136 | `/teams-channel <topic>` | `/teams-kanal <thema>` | Lookup | Find the right MS Teams channel |
| 137 | `/remote` | `/fernzugriff` | Lookup | Remote desktop / home office setup |
| 138 | `/backup` | `/backup` | Lookup | How to back up my files |
| 139 | `/monitor` | `/bildschirm` | Action | Request equipment (monitor, headset, keyboard...) |
| 140 | `/sap <tcode>` | `/sap <tcode>` | Lookup | ERP transaction code description & guide |
| 141 | `/sap-help <topic>` | `/sap-hilfe <thema>` | Lookup | ERP how-to for common tasks |
| 142 | `/erp` | `/erp` | Open | Open ERP web interface |
| 143 | `/sharepoint` | `/sharepoint` | Open | Open department SharePoint |
| 144 | `/intranet` | `/intranet` | Open | Open company intranet |
| 145 | `/onedrive` | `/onedrive` | Open | Open OneDrive |

## 8. HR, Policies & Compliance (146-170)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 146 | `/travel` | `/reise` | Lookup | Travel expense policy summary |
| 147 | `/expense <amount>` | `/spesen <betrag>` | Lookup | Expense limits & how to submit |
| 148 | `/expense-form` | `/reisekosten` | Open | Open travel expense form |
| 149 | `/sick` | `/krank` | Lookup | How to report sick leave |
| 150 | `/parental` | `/elternzeit` | Lookup | Parental leave policy & process |
| 151 | `/homeoffice` | `/homeoffice` | Lookup | Home office policy & rules |
| 152 | `/training <topic>` | `/schulung <thema>` | Lookup | Find training courses & registration |
| 153 | `/payslip` | `/gehaltsabrechnung` | Open | Where to find my pay slip |
| 154 | `/benefits` | `/benefits` | Lookup | Employee benefits overview |
| 155 | `/fleet` | `/fuhrpark` | Lookup | Company car policy & fleet contacts |
| 156 | `/safety` | `/sicherheit` | Lookup | Safety guidelines for my work area |
| 157 | `/works-council` | `/betriebsrat` | Lookup | Works council contacts & office hours |
| 158 | `/suggestion` | `/verbesserung` | Action | Submit an improvement suggestion |
| 159 | `/onboarding` | `/einarbeitung` | Lookup | Onboarding checklist for new employees |
| 160 | `/offboarding` | `/austritt` | Lookup | Offboarding checklist |
| 161 | `/dress-code` | `/kleiderordnung` | Lookup | Dress code by area (office, production, lab) |
| 162 | `/overtime` | `/ueberstunden` | Lookup | Overtime policy & current balance |
| 163 | `/flextime` | `/gleitzeit` | Lookup | Flextime rules & core hours |
| 164 | `/pension` | `/rente` | Lookup | Company pension scheme info |
| 165 | `/insurance` | `/versicherung` | Lookup | Group insurance policies |
| 166 | `/whistleblower` | `/hinweisgeber` | Open | Anonymous reporting channel |
| 167 | `/gdpr` | `/datenschutz` | Lookup | GDPR / data protection guidelines |
| 168 | `/export-control` | `/exportkontrolle` | Lookup | Export control rules & classification |
| 169 | `/anti-bribery` | `/antikorruption` | Lookup | Anti-bribery & gifts policy |
| 170 | `/code-of-conduct` | `/verhaltenskodex` | Open | Open the Code of Conduct |

## 9. Knowledge Base & Documents (171-185)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 171 | `/how <topic>` | `/wie <thema>` | Lookup | Search knowledge base for a process or guide |
| 172 | `/form <name>` | `/formular <name>` | Open | Find and open a form (leave request, expenses...) |
| 173 | `/process <name>` | `/prozess <name>` | Lookup | Process documentation lookup |
| 174 | `/policy <keyword>` | `/richtlinie <stichwort>` | Lookup | Company policy search |
| 175 | `/iso <number>` | `/iso <nummer>` | Lookup | ISO standard / quality document lookup |
| 176 | `/audit` | `/audit` | Lookup | Next audit dates & preparation info |
| 177 | `/sop <keyword>` | `/sop <stichwort>` | Lookup | Standard Operating Procedure search |
| 178 | `/changelog` | `/aenderungen` | Lookup | Recent process/policy changes |
| 179 | `/org-news` | `/unterneuheiten` | Lookup | Latest organizational announcements |
| 180 | `/qa <product>` | `/qa <produkt>` | Lookup | Quality specs & test procedures for a product |
| 181 | `/msds <material>` | `/sdb <werkstoff>` | Open | Material Safety Data Sheet lookup |
| 182 | `/spec <product>` | `/spez <produkt>` | Lookup | Full technical specification |
| 183 | `/standard <din/en>` | `/norm <din/en>` | Lookup | DIN/EN standard description & applicability |
| 184 | `/abbreviation <ABC>` | `/abkuerzung <ABC>` | Lookup | What does this internal acronym mean? |
| 185 | `/glossary` | `/glossar` | Open | Open the company glossary |

## 10. Productivity & Utilities (186-200)

| # | Bolt | German | Type | Description |
|---|-----|--------|------|-------------|
| 186 | `/timer <minutes>` | `/timer <minuten>` | App | Countdown timer mini-app |
| 187 | `/stopwatch` | `/stoppuhr` | App | Stopwatch mini-app |
| 188 | `/pomodoro` | `/pomodoro` | App | Pomodoro focus timer (25min work / 5min break) |
| 189 | `/notes` | `/notizen` | App | Quick scratchpad / sticky notes |
| 190 | `/todo` | `/aufgaben` | App | Personal to-do list mini-app |
| 191 | `/weather` | `/wetter` | Lookup | Weather at current site |
| 192 | `/news` | `/nachrichten` | Lookup | Latest company news / intranet highlights |
| 193 | `/image <prompt>` | `/bild <prompt>` | Prompt | Generate an image with AI |
| 194 | `/whiteboard` | `/whiteboard` | App | Simple drawing / whiteboard mini-app |
| 195 | `/qr <text>` | `/qr <text>` | App | Generate a QR code |
| 196 | `/json` | `/json` | App | JSON formatter / viewer mini-app |
| 197 | `/diff` | `/diff` | App | Text diff / compare tool |
| 198 | `/color <hex>` | `/farbe <hex>` | App | Color picker / palette tool |
| 199 | `/help` | `/hilfe` | Lookup | List all available bolts with descriptions |
| 200 | `/random` | `/zufall` | Lookup | Random useful bolt suggestion ("Did you know...") |

---

## Architecture

Bolts live inside nudges (droplets). Each nudge can define zero or more bolts
in its **Bolts tab** in the nudge editor. This keeps bolt configuration,
permissions, and MCP tool availability tied to the nudge that owns them.

The host app can rename "Bolts" in the UI (e.g. Lechler uses "Jets").

### Bolt Definition (inside a nudge)

Each bolt has:

```python
class BoltDef:
    # Identity
    command: str                    # e.g. "product"
    aliases: list[str] = []        # e.g. ["produkt"] (i18n)
    label: str                     # Display name, e.g. "Product Lookup"
    icon: str = "bolt"             # Material icon name
    description: str = ""
    description_i18n: dict = {}    # {"de": "Produkt nachschlagen", ...}

    # Device compatibility
    devices: list[str] = ["desktop", "tablet", "mobile"]  # show on all by default
    min_width: int | None = None   # alternative: min viewport width in px (e.g. 768)

    # Detection -- how the bolt gets triggered
    # Option A: explicit /command
    # Option B: regex auto-detection (see below)
    regex: str | None = None       # e.g. r"\b\d{3}\.\d{3}\.\d{2}\.\d{2}\b"
    counter_check: str | None = None  # JS or Python expression for 2nd-pass validation

    # Action -- what happens when the bolt fires (extensible)
    action: BoltAction
```

### Bolt Actions (extensible)

Three action types ship out of the box. The system is designed to be extended
with new action types over time.

```python
class BoltAction:
    """Base class. Extend for new action types."""
    type: str  # "url" | "prompt" | "mcp_tool" | future types...

class UrlAction(BoltAction):
    """Open a URL. Supports {input} placeholder."""
    type = "url"
    url: str               # e.g. "https://erp.company.com/product/{input}"
    target: str = "_blank" # "_blank" | "_self"

class PromptAction(BoltAction):
    """Combine user input with a prompt template, send to AI."""
    type = "prompt"
    template: str          # e.g. "Translate the following to {lang}:\n\n{input}"
    auto_send: bool = True # False = pre-fill input, let user edit first

class McpToolAction(BoltAction):
    """Call an MCP tool from the nudge's registered tools."""
    type = "mcp_tool"
    tool_name: str         # e.g. "resolve_contact"
    arg_mapping: dict = {} # e.g. {"name": "{input}"} -- map input to tool args
```

### Two Trigger Modes

#### 1. Explicit: `/command`

User types `/product 123.456.78.99` in the chat input. The input parser
detects the `/` prefix, looks up the bolt, and executes it. Autocomplete
appears on `/` keystroke showing all available bolts with icons and
descriptions.

#### 2. Auto-detect: regex + counter-check

Bolts can passively watch what the user types and surface a **floating
suggestion chip** above the input when they detect a match. This enables
"zero-knowledge" bolt discovery -- users don't need to know bolt commands
exist.

```
┌──────────────────────────────────────────────────┐
│                                                  │
│   ┌──────────────────┐                           │
│   │ ⚡ Product Lookup │  ← floating chip         │
│   └──────────────────┘                           │
│  ┌────────────────────────────────────────────┐  │
│  │ please check availability of 123.456.78.99 │  │
│  └────────────────────────────────────────────┘  │
│                                          [Send]  │
└──────────────────────────────────────────────────┘
```

**Detection pipeline:**

```
User types in input field
       |
       v
  [Regex pass] -- each bolt's regex is tested against the full input text
       |          (debounced, runs on input change)
       | matches
       v
  [Counter-check pass] -- optional second validation
       |  JS:     eval function receives { input, match } → boolean
       |  Python:  server-side call receives { input, match } → boolean
       |
       | returns true
       v
  [Show suggestion chip]
       |
       +---> User clicks chip  ──→ execute bolt action
       +---> User presses Shift+Enter ──→ execute bolt action
       +---> User presses Enter (normal send) ──→ ignore bolt, send to AI
```

**The chip tooltip** reads: `"⚡ Product Lookup — click or Shift+Enter"`
so users learn the keyboard shortcut naturally.

**Multiple matches**: If several bolts match simultaneously, show multiple
chips stacked horizontally. Shift+Enter triggers the first (leftmost) match.

#### Example: Product Number Detection

```python
BoltDef(
    command="product",
    aliases=["produkt"],
    label="Product Lookup",
    icon="inventory_2",
    regex=r"\b\d{3}\.\d{3}\.\d{2}\.\d{2}\b",
    counter_check=None,  # regex is sufficient
    action=McpToolAction(
        tool_name="product_lookup",
        arg_mapping={"product_number": "{match}"},
    ),
)
```

#### Example: Math Expression Detection

```python
BoltDef(
    command="calc",
    aliases=["rechner"],
    label="Calculate",
    icon="calculate",
    regex=r"[0-9]+\s*[\+\-\*\/\^]\s*[0-9]",  # basic math pattern
    counter_check="(match) => { try { return !isNaN(eval(match[0])); } catch { return false; } }",
    action=PromptAction(
        template="Calculate: {input}",
        auto_send=True,
    ),
)
```

### Nudge Editor: Bolts Tab

The nudge editor gains a new **Bolts** tab (alongside General, Files,
Tools, etc.) where admins configure bolts visually:

| Field | Input | Notes |
|-------|-------|-------|
| Command | text | e.g. `product` (the `/product` part) |
| Aliases | chips | e.g. `produkt`, `artikel` |
| Label | text | Display name in chip & autocomplete |
| Icon | icon picker | Material icon |
| Devices | checkboxes | Desktop / Tablet / Mobile (all checked by default) |
| Regex | text | Optional auto-detect pattern |
| Counter-check | code editor | Optional JS or Python expression |
| Action type | dropdown | URL / Prompt / MCP Tool |
| Action config | dynamic form | URL field, template editor, or tool picker |

The tool picker for MCP Tool actions only shows tools that the current
nudge has registered -- bolts can only call tools their nudge owns.

### Device Compatibility

Not all bolts make sense on all devices. A whiteboard mini-app needs a
large screen; a QR code scanner needs a camera. Bolts declare which
devices they support via `devices` (or `min_width` for finer control).

**Device classification** (client-side, based on viewport width):

| Device | Viewport |
|--------|----------|
| `mobile` | < 768px |
| `tablet` | 768px – 1023px |
| `desktop` | >= 1024px |

**Filtering applies everywhere:**
- Autocomplete overlay: incompatible bolts are hidden
- Regex auto-detection: incompatible bolts don't run
- Auto-revive: incompatible bolts show a **chip** instead of a window
  (e.g. "Timer — 12:34 remaining" chip on mobile instead of a floating
  window), or are skipped entirely if the bolt opts out

**Revive fallback**: A bolt can specify a `revive.fallback` for
unsupported devices:

```javascript
await context.memory.registerRevive({
  action: 'window',
  label: 'Whiteboard',
  icon: 'draw',
  state: { ... },
  fallback: 'chip',    // on mobile: show chip instead of window
  // fallback: 'none'  // on mobile: don't revive at all
});
```

Default fallback is `'chip'` — so a timer started on desktop still shows
its countdown as a chip on mobile.

### Client-Side Components

1. **Autocomplete overlay** -- appears on `/` keystroke, fuzzy-matches bolt
   commands and aliases, shows icon + description. Grouped by nudge.
2. **Suggestion chips** -- floating above the input field when regex
   auto-detection matches. Horizontally stacked, clickable, with
   Shift+Enter tooltip.
3. **Bolt result rendering** -- reuses existing infrastructure:
   - URL actions: `window.open(url)`
   - Prompt actions: inject into chat input or auto-send
   - MCP tool actions: call tool, render result as inline card / rich_mcp block

### Execution Pipeline

```
User input
   |
   +---> starts with "/" ──→ [Autocomplete + explicit dispatch]
   |
   +---> any text ──→ [Regex engine] ──→ [Counter-check] ──→ [Chip]
                                                                |
                                      click or Shift+Enter ←───┘
                                                |
                                                v
                                      [Action router]
                                                |
                                +───────────────+───────────────+
                                |               |               |
                              url           prompt          mcp_tool
                                |               |               |
                          window.open()   inject/send    call tool → render
```

### Bolt Memory & Auto-Revive

Bolts can persist state and automatically restore themselves across page
reloads, browser restarts, and session changes.

#### Storage

Bolt memory is stored in **MongoDB** (server-side), so state syncs across
devices — a timer started on desktop continues on a smartphone.

```
Collection: bolt_memory

{
  _id: ObjectId,
  user_id: str,                    // owning user
  nudge_uid: str,                  // scoping — no cross-nudge access
  scope: "permanent" | "session",
  session_id: str | null,          // only set for scope="session"
  key: str,
  value: any,                      // JSON-serializable
  updated_at: datetime,
}

Index: (user_id, nudge_uid, scope, key) — unique
TTL index: updated_at on scope="session" docs (e.g. 7 days)
```

Each nudge can only read/write keys under its own `nudge_uid` —
no cross-nudge access. Two scopes:

| Scope | Lifetime | Use case |
|-------|----------|----------|
| **permanent** | Until explicitly deleted | User preferences, saved notes, todo lists |
| **session** | Until session ends or TTL expires | Timer state, in-progress calculations, form drafts |

#### API (available to bolt actions and mini-apps)

The client-side API talks to the server via Socket.IO events
(`bolt:mem:get`, `bolt:mem:set`, etc.), scoped to the authenticated user
and the bolt's nudge_uid.

```javascript
// context object passed to bolt actions and mini-apps
context.memory = {
  // Permanent (nudge-scoped, synced across devices via MongoDB)
  async get(key),                    // → value | null
  async set(key, value),             // value must be JSON-serializable
  async delete(key),
  async keys(),                      // → string[]

  // Session-scoped (auto-prefixed with session ID)
  async sessionGet(key),
  async sessionSet(key, value),
  async sessionDelete(key),
  async sessionKeys(),

  // Revive registration (see below)
  async registerRevive(config),
  async unregisterRevive(),
};
```

#### Auto-Revive

A bolt can register itself for auto-revive. On page load, the bolt
framework checks for active revive registrations and automatically
restores the bolt's UI.

```javascript
// Inside a timer bolt after user starts a 25-minute timer:
await context.memory.registerRevive({
  // How to restore the bolt
  action: 'window',            // 'window' | 'chip' | 'toast'
  label: 'Pomodoro Timer',     // window title
  icon: 'timer',

  // Bolt state to restore (snapshot)
  state: {
    endTime: Date.now() + 25 * 60 * 1000,
    mode: 'focus',
  },
});

// When the timer finishes or user dismisses:
await context.memory.unregisterRevive();
```

**Revive registration** is stored as a special memory key:

```
bolt:<nudge_uid>:__revive__
```

**On page load**, the bolt framework:

```
1. Fetch all revive entries from MongoDB for this user
2. For each revive entry:
   a. Find the owning nudge (by uid)
   b. Check if the nudge is still available to the user
   c. Check device compatibility (skip if current device not in bolt.devices)
   d. Resolve the bolt definition
   e. Execute the bolt's render with the saved state
   f. Open as floating window / chip / toast (per revive config)
```

**Revive actions:**

| Action | Behavior |
|--------|----------|
| `window` | Open as floating, draggable preview window (timer, calculator, notes) |
| `chip` | Show a persistent chip above the input ("Timer: 12:34 remaining") |
| `toast` | Show a notification toast ("Your timer finished!") |

#### Example: Timer with Auto-Revive

```javascript
// bolt-apps/timer.js (mini-app rendered inside a bolt)
export default {
  name: 'timer',

  async render(container, args, context) {
    // Check if restoring from revive
    const revive = context.reviveState;  // populated if auto-revived
    const endTime = revive?.endTime ?? Date.now() + (args.minutes ?? 5) * 60000;

    // Register for revive (survives reload)
    await context.memory.registerRevive({
      action: 'window',
      label: `Timer — ${args.minutes ?? 5}m`,
      icon: 'timer',
      state: { endTime },
    });

    // Render countdown UI...
    const display = document.createElement('div');
    container.appendChild(display);

    const tick = () => {
      const remaining = endTime - Date.now();
      if (remaining <= 0) {
        display.textContent = '00:00 — Done!';
        context.memory.unregisterRevive();
        return;
      }
      const m = Math.floor(remaining / 60000);
      const s = Math.floor((remaining % 60000) / 1000);
      display.textContent = `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
      requestAnimationFrame(tick);
    };
    tick();
  },
};
```

#### Mini-App Framework (updated)

Mini-apps are self-contained JS modules rendered as doc plugin blocks.
They receive a `context` object with memory, dark mode, locale, and
revive state:

```javascript
export default {
  name: 'calculator',
  render(container, args, context) {
    // context.isDark       — boolean
    // context.locale       — 'en' | 'de' | ...
    // context.close()      — close the window
    // context.memory       — BoltMemory API (scoped to this nudge)
    // context.reviveState  — saved state if auto-revived, else null
  },
};
```

Mini-apps render in the existing doc plugin infrastructure and can be:
- Shown inline in chat
- Opened in floating, draggable, resizable preview windows
- Maximized to full screen
- **Auto-revived** on page load if registered

### Implementation Priority

**Tier 1 -- Wire existing nudge backends**

| Bolts | Nudge / Backend |
|-------|-----------------|
| `/who`, `/phone`, `/email`, `/org` | People nudge → O365 MCP tools |
| `/product`, `/customer`, `/debtor` | Sales nudge → Product/Customer DB MCP |
| `/lunch` | Canteen nudge → menu service |
| `/today`, `/week`, `/next`, `/free` | Calendar nudge → O365 calendar tools |
| `/birthday`, `/jubilee` | Celebrations nudge → celebrations service |
| `/translate`, `/formal`, `/summarize` | Text nudge → prompt actions |

**Tier 2 -- URL redirects & prompt templates (no backend needed)**

| Bolts | Action |
|-------|--------|
| `/erp`, `/sharepoint`, `/intranet` | URL actions |
| `/catalog`, `/map`, `/payslip` | URL actions |
| `/quotation`, `/visit`, `/template` | Prompt actions |
| `/proofread`, `/shorten`, `/bullet` | Prompt actions |

**Tier 3 -- Regex auto-detection bolts**

| Bolts | Pattern | Action |
|-------|---------|--------|
| Product number | `\d{3}\.\d{3}\.\d{2}\.\d{2}` | MCP tool lookup |
| Math expression | `[0-9]+\s*[+\-*/^]\s*[0-9]` | Prompt → Math nudge |
| Email address | `[\w.]+@[\w.]+` | MCP tool → contact lookup |
| Order number | `[A-Z]{2}-\d{6,}` | MCP tool → order status |

**Tier 4 -- New backend integrations**

| Bolts | Requires |
|-------|----------|
| `/order`, `/stock`, `/leadtime` | ERP/SAP API |
| `/ticket`, `/tickets` | IT ticketing system API |
| `/pto`, `/overtime` | HR system API |
| `/room`, `/book` | Exchange room mailbox integration |
| `/fx` | Currency rate API |
