"""
Builds the 200-article Member 4 sample dataset (/ai/data/sample.json) and the
labeled evaluation datasets (/ai/data/eval_classification_50.json and
/ai/data/eval_clickbait_30.json).

Run once or whenever refreshing /ai/data/*.json:
    python3 ai/data/build_sample_data.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

BASE_TIME = datetime(2026, 11, 12, 18, 0, 0, tzinfo=timezone.utc)

# 20 multi-source news stories (each expanded across 3-4 independent outlets within 48h -> ~70 clustered articles)
MULTI_SOURCE_CLUSTERS = [
    {
        "cluster_id": "clu_isro_ce20",
        "topic": "science",
        "language": "en",
        "hours_ago": 4,
        "core": (
            "ISRO completes 200-second CE-20 cryogenic engine hot test at Mahendragiri",
            "The Indian Space Research Organisation successfully conducted a 200-second flight-acceptance hot test of the upgraded CE-20 cryogenic engine at its Mahendragiri propulsion complex in Tamil Nadu. Engineers verified throttle modulation and chamber pressure stability ahead of upcoming Gaganyaan and heavy-lift LVM3 commercial missions."
        ),
        "variants": [
            ("The Hindu", "ISRO Completes 200-Second Cryogenic Engine Hot Test at Mahendragiri"),
            ("Deccan Herald", "Mahendragiri Facility Clears CE-20 Cryogenic Test for Upcoming LVM3 Flight"),
            ("Indian Express", "ISRO Validates Upgraded CE-20 Cryogenic Stage in Key Hot Fire Test"),
            ("Reuters", "India's Space Agency Passes Crucial Cryogenic Propulsion Milestone"),
        ],
    },
    {
        "cluster_id": "clu_rbi_upi",
        "topic": "business",
        "language": "en",
        "hours_ago": 7,
        "core": (
            "RBI expands UPI Lite wallet limits and offline digital payment caps",
            "The Reserve Bank of India raised the per-transaction limit for UPI Lite offline wallets to Rs 1,000 and total wallet balance to Rs 5,000 to accelerate low-value retail payments in semi-urban corridors without PIN friction."
        ),
        "variants": [
            ("Mint", "RBI Raises UPI Lite Offline Transaction Cap to Rs 1,000"),
            ("Economic Times", "Central Bank Boosts UPI Lite Wallet Ceiling to Rs 5,000 for Faster Retail Payments"),
            ("Business Standard", "RBI Expands Offline Digital Payment Limits Under UPI Lite Framework"),
        ],
    },
    {
        "cluster_id": "clu_blr_metro",
        "topic": "india",
        "language": "en",
        "hours_ago": 10,
        "core": (
            "Bengaluru Metro Yellow Line completes final signaling safety trials",
            "Bangalore Metro Rail Corporation Limited wrapped up driverless CBTC signaling trials along the 19-km RV Road to Bommasandra Yellow Line corridor, clearing the path for Commissioner of Metro Railway Safety inspection this month."
        ),
        "variants": [
            ("Deccan Herald", "Bengaluru Metro Yellow Line Completes Final Signaling Safety Trials"),
            ("The Hindu", "BMRCL Wraps Up Driverless CBTC Testing on Electronics City Corridor"),
            ("Indian Express", "Yellow Line Metro Awaits Final Safety Nod After Signaling Trials"),
        ],
    },
    {
        "cluster_id": "clu_ai_chips",
        "topic": "technology",
        "language": "en",
        "hours_ago": 12,
        "core": (
            "Open-source edge AI chip alliance targets sub-5W inference modules",
            "A consortium of semiconductor startups and university labs unveiled an open RISC-V neural accelerator architecture delivering 40 TOPS under a 5-watt envelope for smartphones, drones, and offline translation devices."
        ),
        "variants": [
            ("TechCrunch", "Semiconductor Alliance Unveils Sub-5W Open RISC-V Edge AI Accelerator"),
            ("Ars Technica", "New Open Neural Chip Design Promises 40 TOPS Under Five Watts"),
            ("Mint", "Edge AI Chip Consortium Targets Low-Power On-Device Inference"),
            ("Reuters", "Chip Startups Back Open RISC-V Standard for Mobile AI Processing"),
        ],
    },
    {
        "cluster_id": "clu_ind_cricket",
        "topic": "sports",
        "language": "en",
        "hours_ago": 14,
        "core": (
            "India pacers seal 48-run victory in Perth series opener",
            "India's pace attack claimed seven wickets in the final session at Perth to wrap up a 48-run win in the opening Test match, backed by a disciplined fourth-innings defense and sharp slip catching."
        ),
        "variants": [
            ("ESPNcricinfo", "India Pacers Strike Late to Seal 48-Run Victory in Perth Opener"),
            ("The Hindu", "Clinical Pace Bowling Gives India 1-0 Series Lead in Perth"),
            ("Indian Express", "Seamers Deliver Memorable Final-Session Win for India at Perth"),
        ],
    },
    {
        "cluster_id": "clu_cop_climate",
        "topic": "world",
        "language": "en",
        "hours_ago": 18,
        "core": (
            "Global climate negotiators agree on grid-storage financing framework",
            "Delegates from 140 countries reached consensus on a concessional clean-grid and battery storage financing facility aimed at helping emerging economies integrate solar and wind generation before 2030."
        ),
        "variants": [
            ("Reuters", "Climate Negotiators Back Concessional Grid-Storage Fund for Emerging Markets"),
            ("BBC", "140 Nations Agree on Clean Grid Battery Financing Framework"),
            ("The Hindu", "Global Summit Unlocks New Capital for Renewable Grid Storage"),
        ],
    },
    {
        "cluster_id": "clu_dengue_vaccine",
        "topic": "health",
        "language": "en",
        "hours_ago": 20,
        "core": (
            "Phase 3 dengue vaccine trial reports 81 percent efficacy across multi-city cohort",
            "Researchers running a 10,000-participant Phase 3 clinical trial across 18 Indian medical centers reported 81 percent overall efficacy against symptomatic dengue across all four serotypes with no severe safety signals."
        ),
        "variants": [
            ("The Hindu", "Phase 3 Dengue Vaccine Trial Shows 81 Percent Efficacy Across 18 Centers"),
            ("Indian Express", "Multi-City Clinical Study Delivers Promising Dengue Vaccine Results"),
            ("BBC", "New Single-Dose Dengue Candidate Achieves Strong Phase 3 Protection"),
        ],
    },
    {
        "cluster_id": "clu_nep_credits",
        "topic": "education",
        "language": "en",
        "hours_ago": 22,
        "core": (
            "UGC launches national Academic Bank of Credits skill-transfer portal",
            "The University Grants Commission rolled out an updated Academic Bank of Credits portal allowing engineering and humanities undergraduates to transfer verified micro-credentials and vocational internships across 420 universities."
        ),
        "variants": [
            ("Indian Express", "UGC Expands Academic Bank of Credits to 420 Universities"),
            ("Deccan Herald", "Students Can Now Transfer Micro-Credentials Nationwide via UGC Portal"),
            ("The Hindu", "Updated Academic Credit Bank Integrates Vocational Internships"),
        ],
    },
    {
        "cluster_id": "clu_iffi_cinema",
        "topic": "entertainment",
        "language": "en",
        "hours_ago": 25,
        "core": (
            "Independent regional cinema dominates Goa international film festival slate",
            "Twelve debut directors from Kannada, Malayalam, Marathi, and Assamese cinema secured premiere slots in the Indian Panorama competition, highlighting a shift toward small-budget character dramas."
        ),
        "variants": [
            ("The Hindu", "Regional Indie Filmmakers Lead Indian Panorama Lineup in Goa"),
            ("Deccan Herald", "Kannada and Malayalam Debuts Shine in Film Festival Selection"),
            ("Indian Express", "Independent Regional Cinema Takes Center Stage at Annual Film Festival"),
        ],
    },
    {
        "cluster_id": "clu_quantum_fiber",
        "topic": "technology",
        "language": "en",
        "hours_ago": 27,
        "core": (
            "Telecom lab demonstrates quantum key distribution over 180 km commercial fiber",
            "Researchers successfully transmitted tamper-evident quantum encryption keys across 180 kilometers of live metropolitan fiber-optic cable without cryogenic repeaters, bringing quantum-safe banking links closer to deployment."
        ),
        "variants": [
            ("Ars Technica", "Quantum Key Distribution Runs Across 180 km of Commercial Metro Fiber"),
            ("TechCrunch", "Telecom Researchers Achieve 180-km Quantum-Safe Fiber Link"),
            ("Mint", "Quantum Encryption Test Over Live Fiber Sets Domestic Distance Record"),
        ],
    },
    {
        "cluster_id": "clu_ev_battery",
        "topic": "business",
        "language": "en",
        "hours_ago": 29,
        "core": (
            "Domestic lithium-iron-phosphate cell plant begins commercial shipments",
            "A 10 GWh lithium-iron-phosphate battery gigafactory in Gujarat commenced commercial cell deliveries to two-wheeler and compact electric car makers, trimming imported pack costs by nearly 14 percent."
        ),
        "variants": [
            ("Mint", "Gujarat LFP Battery Gigafactory Begins Commercial Cell Shipments"),
            ("Business Standard", "Local EV Cell Manufacturing Cuts Battery Pack Costs by 14 Percent"),
            ("Economic Times", "10 GWh Domestic LFP Plant Starts Supplying Indian EV Automakers"),
        ],
    },
    {
        "cluster_id": "clu_jwst_exoplanet",
        "topic": "science",
        "language": "en",
        "hours_ago": 31,
        "core": (
            "Webb telescope detects water vapor and methane signatures on temperate exoplanet",
            "Astronomers analyzing James Webb Space Telescope transmission spectra confirmed clear signatures of water vapor and atmospheric methane on a sub-Neptune exoplanet 110 light-years away."
        ),
        "variants": [
            ("ScienceDaily", "James Webb Telescope Detects Water Vapor and Methane on Nearby Exoplanet"),
            ("BBC", "Webb Spectra Reveal Atmospheric Water Signatures 110 Light-Years Away"),
            ("Reuters", "Astronomers Spot Methane and Water Vapor in Temperate Exoplanet Atmosphere"),
        ],
    },
    {
        "cluster_id": "clu_monsoon_ai",
        "topic": "science",
        "language": "en",
        "hours_ago": 33,
        "core": (
            "Meteorological department deploys graph neural weather model for district rainfall",
            "The meteorological agency introduced a high-resolution graph neural forecasting pipeline capable of generating 72-hour district-level rainfall alerts in under six minutes on regional GPU clusters."
        ),
        "variants": [
            ("The Hindu", "Weather Agency Deploys Graph Neural Model for District Rainfall Alerts"),
            ("Indian Express", "AI Weather Pipeline Cuts 72-Hour Monsoon Forecast Compute to Six Minutes"),
            ("Deccan Herald", "High-Resolution Neural Weather Model Enters Operational Testing"),
        ],
    },
    {
        "cluster_id": "clu_women_football",
        "topic": "sports",
        "language": "en",
        "hours_ago": 35,
        "core": (
            "National women's football league expands to 12 clubs with youth academy mandate",
            "The football federation approved expansion of the top-tier women's league to 12 clubs starting next season, requiring every franchise to operate U-17 regional development academies."
        ),
        "variants": [
            ("The Hindu", "Women's Football League Expands to 12 Clubs With Youth Academy Rule"),
            ("Indian Express", "Football Federation Clears 12-Team Top Tier Women's Season"),
            ("ESPNcricinfo", "Expanded Women's League Mandates U-17 Grassroots Academies"),
        ],
    },
    {
        "cluster_id": "clu_eu_cable",
        "topic": "world",
        "language": "en",
        "hours_ago": 37,
        "core": (
            "European maritime patrol coalition launches subsea internet cable monitoring",
            "Twelve coastal nations initiated joint autonomous underwater drone patrols to monitor subsea fiber-optic cables and offshore energy interconnectors across the North and Baltic seas."
        ),
        "variants": [
            ("Reuters", "European Nations Launch Joint Drone Patrols for Subsea Internet Cables"),
            ("BBC", "Twelve Coastal States Coordinate Protection of Undersea Fiber Links"),
            ("Ars Technica", "Maritime Coalition Deploys Autonomous Sensors Along Subsea Cables"),
        ],
    },
    {
        "cluster_id": "clu_telemedicine",
        "topic": "health",
        "language": "en",
        "hours_ago": 39,
        "core": (
            "Rural telemedicine network crosses 250 million primary consultations",
            "Public primary health centers connected to district specialists via low-bandwidth teleconsultation hubs surpassed 250 million cumulative patient visits, cutting specialist travel time in hill districts."
        ),
        "variants": [
            ("The Hindu", "Public Telemedicine Network Crosses 250 Million Rural Consultations"),
            ("Deccan Herald", "Low-Bandwidth Telehealth Hubs Expand Specialist Care in Rural Districts"),
            ("Mint", "Digital Primary Health Network Hits 250 Million Patient Visits"),
        ],
    },
    {
        "cluster_id": "clu_coding_olympiad",
        "topic": "education",
        "language": "en",
        "hours_ago": 41,
        "core": (
            "Public school students sweep national robotics and algorithms olympiad",
            "Teams from government residential schools in Karnataka, Odisha, and Maharashtra claimed four of the top six medals at the National School Robotics and Algorithms Olympiad in Pune."
        ),
        "variants": [
            ("Deccan Herald", "Government School Teams Win Top Honors at National Robotics Olympiad"),
            ("The Hindu", "Karnataka and Odisha Students Shine in Pune Algorithms Finals"),
            ("Indian Express", "Public School Coders Take Four of Six Medals at Robotics Olympiad"),
        ],
    },
    {
        "cluster_id": "clu_semicon_freight",
        "topic": "india",
        "language": "en",
        "hours_ago": 43,
        "core": (
            "Dedicated freight corridor cuts container transit time between Gujarat and NCR",
            "Western Dedicated Freight Corridor operations reduced average double-stack container transit times between Mundra port and the National Capital Region from 68 hours to 26 hours."
        ),
        "variants": [
            ("Mint", "Dedicated Freight Corridor Cuts Port-to-NCR Cargo Transit to 26 Hours"),
            ("Business Standard", "Western Rail Freight Corridor Doubles Average Container Speeds"),
            ("The Hindu", "Double-Stack Electric Freight Trains Slash Logistics Delays"),
        ],
    },
    {
        "cluster_id": "clu_audio_books",
        "topic": "entertainment",
        "language": "en",
        "hours_ago": 44,
        "core": (
            "Bilingual audiobook publishers report surge in regional mystery podcasts",
            "Audio streaming platforms noted a 64 percent year-on-year rise in Kannada, Hindi, and Tamil serialized detective fiction, driven by daily urban metro commuters."
        ),
        "variants": [
            ("Mint", "Regional Mystery Audiobooks Post 64 Percent Listener Growth"),
            ("Deccan Herald", "Commuters Drive Surge in Kannada and Hindi Audio Fiction"),
            ("The Hindu", "Serialized Detective Podcasts Gain Momentum Across Indian Languages"),
        ],
    },
    {
        "cluster_id": "clu_opensource_llm",
        "topic": "technology",
        "language": "en",
        "hours_ago": 45,
        "core": (
            "Researchers release open multilingual benchmark for Indic document OCR",
            "Academic NLP labs published an open 120,000-page document OCR and summarization benchmark covering 14 Indian scripts including Kannada, Devanagari, Tamil, and Bengali."
        ),
        "variants": [
            ("TechCrunch", "Open Multilingual Benchmark Released for 14 Indian Script Documents"),
            ("Ars Technica", "Academic Labs Publish 120,000-Page Indic OCR Evaluation Suite"),
            ("Deccan Herald", "New Open Benchmark Targets Kannada and Hindi Document AI Accuracy"),
        ],
    },
]

# Standalone single-source articles across all 9 topics + Kannada + Hindi + Clickbait examples
STANDALONE_ARTICLES = [
    # Technology
    ("technology", "en", "TechCrunch", "WebGPU Adoption Crosses 75 Percent Across Desktop Browsers", "Browser vendors have stabilized WebGPU compute shaders across major desktop platforms, allowing local neural inference and 3D CAD tools to run directly inside browser tabs without native plugins.", 0.06, 4),
    ("technology", "en", "Ars Technica", "Linux Kernel 6.12 Brings Real-Time Scheduling to Mainline Builds", "After two decades of out-of-tree patches, real-time PREEMPT_RT scheduling support has officially landed in the mainline Linux kernel for industrial robotics and audio workstations.", 0.05, 5),
    ("technology", "en", "Mint", "Cloud Firms Shift Inference Workloads to Custom Arm Servers", "Enterprise SaaS providers are migrating stateless inference APIs onto Arm-based server instances to curb datacenter power draw during peak daytime traffic.", 0.08, 3),
    ("technology", "en", "Deccan Herald", "Bengaluru Cybersecurity Lab Flags Malicious Browser Extension Campaign", "Security researchers documented a credential-harvesting campaign disguised as PDF converter extensions affecting 90,000 users before store takedowns.", 0.14, 4),
    ("technology", "en", "TechCrunch", "Local-First Sync Engines Gain Traction Among Field Engineering Apps", "CRDT-based local-first databases are replacing fragile online-only forms for construction and utility inspection crews working in dead cellular zones.", 0.07, 4),
    ("technology", "en", "Ars Technica", "PostgreSQL 18 Preview Improves Asynchronous Disk I/O Throughput", "Early benchmarks on the upcoming PostgreSQL release show up to 35 percent faster sequential scan throughput on NVMe storage arrays.", 0.04, 3),
    ("technology", "en", "Mint", "Open-Source Maintainers Pilot Automated SBOM Attestation Pipelines", "Package registries are rolling out cryptographic provenance checks so enterprise teams can verify build integrity before deploying dependencies.", 0.05, 3),
    ("technology", "en", "The Hindu", "State Digital Mission Standardizes Verifiable Land Record APIs", "Digitized survey maps and encumbrance certificates are being linked under standardized APIs to reduce property registration fraud.", 0.09, 4),
    ("technology", "en", "TechCrunch", "Solid-State Cooling Chips Target Thin Fanless Laptops", "Piezoelectric micro-air-jet modules promise silent active cooling inside 10-millimeter ultrabooks without spinning mechanical bearings.", 0.11, 3),
    ("technology", "en", "Ars Technica", "Passkey Adoption Cuts Account Takeover Tickets by Half at Major Banks", "Financial apps migrating away from SMS one-time codes to FIDO2 passkeys report a sharp drop in phishing-related support escalations.", 0.07, 4),

    # Sports
    ("sports", "en", "ESPNcricinfo", "Ranji Trophy Introduces Smart Ball Tracking for Marginal LBW Reviews", "Domestic cricket umpires are testing automated seam-tracking replays in select Ranji Trophy quarterfinals to reduce review turnaround times.", 0.08, 3),
    ("sports", "en", "The Hindu", "Badminton Federation Tests 15-Point Three-Game Scoring Trial", "International badminton officials are piloting shorter 15-point sets in junior tournaments to reduce match fatigue during packed calendar weeks.", 0.09, 3),
    ("sports", "en", "Indian Express", "Chennai Chess Academy Hosts FIDE Grandmaster Norm Invitational", "Twelve teenage international masters are competing in a 10-round classical event aimed at unlocking final Grandmaster norms on home soil.", 0.05, 3),
    ("sports", "en", "BBC", "Premier League Approves Semi-Automated Offside Graphics Rollout", "Stadium screens and broadcast feeds will display 3D limb-tracking offside visualizations within 30 seconds of tight VAR checks.", 0.10, 3),
    ("sports", "en", "Deccan Herald", "Karnataka Swimmers Sweep Relay Golds at National Aquatics Meet", "Bengaluru-trained relay squads broke two meet records in the 4x100m freestyle finals at the national aquatics championships.", 0.06, 2),
    ("sports", "en", "ESPNcricinfo", "Women's Premier League Announces Expanded Five-Venue Caravan Schedule", "Next season's franchise tournament will add two regional stadiums to widen grassroots attendance beyond metro centers.", 0.07, 3),
    ("sports", "en", "The Hindu", "Archery High-Performance Camp Integrates Wind-Tunnel Arrow Testing", "National compound and recurve archers are calibrating fletching profiles with aerodynamic telemetry ahead of the Asian Championships.", 0.08, 3),
    ("sports", "en", "Indian Express", "Javelin Biomechanics Lab Opens at Patiala Sports Complex", "Motion-capture release sensors are helping track-and-field athletes optimize elbow angles and reduce shoulder strain.", 0.06, 3),
    ("sports", "en", "BBC", "Formula One Teams Finalize 2026 Active Aerodynamics Wind-Tunnel Limits", "Engineers agreed on standardized movable front and rear wing actuation rules designed to improve overtaking efficiency.", 0.09, 4),
    ("sports", "en", "Mint", "Marathon Organizers Mandate Heat-Index Pacemaker Checkpoints", "Major city road races are adding wet-bulb globe temperature monitors every five kilometers to safeguard amateur runners.", 0.05, 2),

    # Business
    ("business", "en", "Mint", "Manufacturing PMI Holds Steady at 57.4 on Export Order Growth", "New export orders in electronics assembly and precision engineering kept India's manufacturing purchasing managers' index comfortably in expansion territory.", 0.05, 3),
    ("business", "en", "Business Standard", "SaaS Exporters Hedge Currency Swings With Multi-Region Billing", "Mid-market software firms are pricing enterprise contracts in local currencies while consolidating cloud spend across regional data centers.", 0.07, 3),
    ("business", "en", "Economic Times", "Port Operators Automate Yard Cranes to Cut Vessel Turnaround", "Container terminals in Nhava Sheva and Krishnapatnam deployed optical container scanners that shave four hours off average berth stays.", 0.06, 3),
    ("business", "en", "Reuters", "Global Copper Inventories Tighten Amid Grid Expansion Demand", "Transformer manufacturers and renewable developers face longer lead times as refined copper stockpiles hit multi-month lows.", 0.08, 4),
    ("business", "en", "Mint", "Mutual Fund SIP Inflows Cross Rs 25,000 Crore Monthly Mark", "Retail systematic investment plan accounts registered record monthly inflows led by first-time investors from tier-2 cities.", 0.07, 3),
    ("business", "en", "Business Standard", "Cold-Chain Logistics Startups Target Floriculture and Dairy Exports", "Solar-assisted modular cold rooms near highway junctions are cutting post-harvest spoilage by 22 percent for cooperative dairies.", 0.06, 3),
    ("business", "en", "Economic Times", "Textile Hubs Adopt Zero-Liquid-Discharge Dyeing Units", "Tiruppur apparel exporters investing in closed-loop water recycling are securing long-term European retail compliance contracts.", 0.05, 4),
    ("business", "en", "Reuters", "Airlines Order Fuel-Efficient Narrowbody Jets for Regional Routes", "Regional carriers signed lease agreements for 45 next-generation turboprop and narrowbody aircraft to connect underserved tier-3 airports.", 0.08, 3),
    ("business", "en", "Mint", "Agri-Fintech Platforms Expand Satellite Crop-Yield Underwriting", "Rural lenders are combining multi-spectral satellite imagery with soil moisture logs to approve seasonal working capital within 48 hours.", 0.07, 4),
    ("business", "en", "Business Standard", "Commercial Real Estate Leasing Surges in GCC Engineering Hubs", "Global capability centers leased 18 million square feet across Bengaluru, Hyderabad, and Pune for R&D and embedded systems teams.", 0.06, 3),

    # Science
    ("science", "en", "ScienceDaily", "Deep-Sea Submersible Maps Hydrothermal Vent Ecosystems in Indian Ocean", "Oceanographers cataloged 14 previously undocumented species of mineral-processing microbes around deep hydrothermal chimneys.", 0.06, 4),
    ("science", "en", "Nature Briefing", "Perovskite-Silicon Tandem Solar Cell Passes 1,000-Hour Damp-Heat Test", "Materials scientists stabilized moisture-resistant encapsulation layers, pushing tandem solar module efficiency past 31 percent in outdoor trials.", 0.05, 4),
    ("science", "en", "The Hindu", "Western Ghats Botanical Survey Identifies Three Endemic Moss Species", "Field botanists documented three rare bryophytes thriving along high-altitude cloud forest streams in Kodagu and Wayanad.", 0.04, 3),
    ("science", "en", "BBC", "Antarctic Ice-Core Drilling Recovers 1.2-Million-Year Climate Record", "Glaciologists extracted continuous deep ice layers preserving ancient atmospheric carbon bubbles that illuminate Pleistocene glacial cycles.", 0.07, 4),
    ("science", "en", "ScienceDaily", "Enzyme Engineering Doubles Biodegradable Plastic Breakdown Speed", "Directed evolution produced a heat-tolerant hydrolase enzyme capable of depolymerizing packaging films within 36 hours in industrial composters.", 0.08, 3),
    ("science", "en", "Reuters", "European Space Agency Tests Autonomous Lunar Rover Navigation", "Prototype rovers traversed volcanic basalt testbeds using stereo vision and onboard hazard mapping without Earth-based joystick control.", 0.06, 3),
    ("science", "en", "Deccan Herald", "Bengaluru Radio Telescope Captures Fast Radio Burst Polarization Shift", "Astrophysicists used upgraded receiver arrays to trace magnetic field reversals around a repeating fast radio burst source.", 0.07, 4),
    ("science", "en", "Nature Briefing", "Urban Heat-Island Study Recommends Reflective Cool-Roof Corridors", "Satellite thermal mapping across six megacities demonstrated that high-albedo coatings lower peak indoor afternoon temperatures by 2.4 degrees Celsius.", 0.05, 4),
    ("science", "en", "ScienceDaily", "Honeybee Waggle-Dance Acoustics Decoded Using Tiny Optical Sensors", "Biologists measured hive vibration frequencies to understand how foragers calibrate floral distance estimates in gusty winds.", 0.09, 3),
    ("science", "en", "The Hindu", "National Gene Bank Expands Drought-Tolerant Millet Seed Vault", "Agronomists cataloged 1,800 traditional finger millet and sorghum landraces in cold storage to safeguard arid-zone food security.", 0.05, 3),

    # India
    ("india", "en", "The Hindu", "Chennai Desalination Plant Adds Solar Microgrid to Cut Power Costs", "Municipal water engineers paired coastal reverse-osmosis desalination pumps with a dedicated solar array to stabilize urban drinking water supply.", 0.06, 3),
    ("india", "en", "Indian Express", "Railways Completes Electrification of Mountain Gauge Branch Lines", "High-torque electric locomotives have replaced diesel haulage on steep gradient hill corridors, cutting fuel expenses and tunnel emissions.", 0.07, 3),
    ("india", "en", "Deccan Herald", "Bengaluru Civic Body Maps 42 Lakes for Real-Time Dissolved Oxygen Sensors", "Automated floating buoys will transmit water quality readings every 15 minutes to alert maintenance crews about sewage inflow.", 0.08, 3),
    ("india", "en", "The Hindu", "Postal Department Expands Digital Address PIN Pilot in 12 Cities", "Grid-coded digital address pins are being tested with emergency ambulances and e-commerce couriers to improve last-mile navigation.", 0.06, 3),
    ("india", "en", "Mint", "Highway Authority Tests Recycled Plastic-Bitumen Resurfacing", "National highway stretches near Pune and Surat are evaluating shredded waste-plastic asphalt blends for monsoon pothole resistance.", 0.09, 3),
    ("india", "en", "Indian Express", "Heritage Conservationists Restore 17th-Century Stepwells in Gujarat", "Lime-mortar masonry repairs and rainwater recharge channels have revived seven historic community stepwells before the summer dry season.", 0.05, 4),
    ("india", "en", "Deccan Herald", "Kochi Water Metro Adds Four Battery-Electric Catamarans", "Zero-emission aluminum ferries equipped with rapid pier chargers have crossed three million passenger trips across island terminals.", 0.06, 3),
    ("india", "en", "The Hindu", "Consumer Commission Mandates Clear Repairability Scores on Appliances", "Draft standards propose 1-to-10 repairability labels on refrigerators, washing machines, and smartphones sold online.", 0.08, 3),
    ("india", "en", "Indian Express", "Northeast Bamboo Bridges Replaced With Modular Composite Spans", "Lightweight fiber-reinforced footbridges now provide all-weather school access across seasonal hill streams in Arunachal Pradesh.", 0.05, 3),
    ("india", "en", "Deccan Herald", "Karnataka Solar Pump Subsidy Reaches 35,000 Smallholder Farms", "Daytime solar irrigation feeders have helped farmers eliminate midnight grid rationing cycles.", 0.07, 3),

    # World
    ("world", "en", "Reuters", "Southeast Asian Grid Pact Links Cross-Border Hydro and Solar Power", "Energy ministers signed bilateral wheeling agreements to balance dry-season hydro deficits with neighboring solar generation.", 0.06, 4),
    ("world", "en", "BBC", "Nordic Cities Expand Underground District Heating Heat-Pump Networks", "Municipal utilities are capturing waste heat from wastewater treatment plants and datacenters to warm residential apartment blocks.", 0.05, 3),
    ("world", "en", "Reuters", "UN Maritime Agency Adopts Stricter Methanol Bunkering Safety Rules", "New international shipping guidelines standardize crew training and leak detection for green-methanol cargo vessels.", 0.07, 3),
    ("world", "en", "BBC", "Tokyo Commuter Rail Tests Regenerative Braking Supercapacitors", "Trackside energy storage banks capture train braking power to accelerate departing expresses during morning rush hour.", 0.06, 3),
    ("world", "en", "The Hindu", "African Continental Free Trade Area Launches Pan-African Payment Rail", "Central banks expanded instant local-currency settlement corridors to cut cross-border SME invoice clearing times from days to seconds.", 0.05, 4),
    ("world", "en", "Reuters", "Panama Canal Water Conservation Measures Restore Daily Transit Slots", "Reservoir management and cross-filling lock chambers allowed authorities to lift draft restrictions ahead of peak shipping season.", 0.07, 3),
    ("world", "en", "BBC", "Global Seed Vault in Svalbard Receives 14,000 New Crop Samples", "Agricultural institutes from 21 countries deposited backup heirloom wheat, barley, and legume varieties into the Arctic permafrost vault.", 0.05, 3),
    ("world", "en", "Reuters", "Latin American Lithium Producers Form Shared Environmental Audit Standard", "Brine extraction operators agreed on public groundwater monitoring dashboards verified by independent hydrogeologists.", 0.08, 4),
    ("world", "en", "BBC", "European Rail Operators Unify Cross-Border Night Train Ticketing", "A shared reservation API now lets travelers book multi-country sleeper itineraries in a single transaction.", 0.06, 3),
    ("world", "en", "The Hindu", "Pacific Island Nations Establish Regional Disaster Relief Drone Hub", "Pre-positioned cargo drones and solar micro-stations will speed medical supply drops after tropical cyclones.", 0.05, 3),

    # Entertainment
    ("entertainment", "en", "The Hindu", "Restored 1960s South Indian Film Classics Set for 4K Theatrical Re-Release", "Film archivists completed frame-by-frame photochemical restoration of six landmark black-and-white dramas using original camera negatives.", 0.05, 3),
    ("entertainment", "en", "Deccan Herald", "Bengaluru Indie Bookstores Launch Late-Night Translation Salons", "Neighborhood bookshops are hosting monthly bilingual readings pairing Kannada short-story writers with English and Hindi translators.", 0.06, 3),
    ("entertainment", "en", "BBC", "Symphony Orchestras Commission Interactive Spatial Audio Scores", "Concert halls in London and Tokyo are experimenting with 64-channel acoustic domes for contemporary orchestral premieres.", 0.07, 4),
    ("entertainment", "en", "Indian Express", "Animation Studios Embrace Hand-Painted Backgrounds With 2D Rigging", "Independent animators are blending traditional gouache textures with modern compositing pipelines for feature storytelling.", 0.06, 3),
    ("entertainment", "en", "The Hindu", "Classical Carnatic Sabhas Live-Stream Archive Concerts for Diaspora", "Digitized concert recordings spanning four decades are being indexed by raga and composer for music students worldwide.", 0.04, 3),
    ("entertainment", "en", "Mint", "Theater Collectives Tour Black-Box Sci-Fi Plays Across College Campuses", "Compact modular stage lighting allows touring repertory groups to perform intimate speculative dramas in university auditoriums.", 0.07, 3),
    ("entertainment", "en", "BBC", "Museum Digitizes 40,000 Historic Jazz Vinyl Liner Notes", "Researchers can now search session personnel, studio dates, and recording engineers across mid-century jazz releases.", 0.05, 3),
    ("entertainment", "en", "Deccan Herald", "Mysuru Craft Guilds Showcase Eco-Friendly Natural Dye Puppetry", "Traditional leather shadow-puppet artists collaborated with designers to bring folk epics to school cultural festivals.", 0.06, 3),

    # Health
    ("health", "en", "The Hindu", "Cardiology Study Links Brisk 20-Minute Post-Meal Walks to Glycemic Control", "A randomized crossover trial across 320 adults showed brief post-dinner walks lowered postprandial glucose spikes by 19 percent.", 0.06, 3),
    ("health", "en", "BBC", "Hospital Sleep Clinics Introduce Non-Invasive Acoustic Apnea Screening", "Bedside smartphone sonar monitoring matched overnight polysomnography accuracy in detecting moderate sleep apnea during clinical validation.", 0.08, 4),
    ("health", "en", "Indian Express", "Public Hospitals Standardize Barcoded Blood-Bank Inventory Tracking", "Real-time cross-district platelet and plasma inventory dashboards reduced emergency blood-unit transfer delays by 40 percent.", 0.05, 3),
    ("health", "en", "Deccan Herald", "Pediatric Nutrition Drive Adds Fortified Millet Mid-Day Meals", "District health audits noted measurable gains in hemoglobin levels six months after introducing iron-rich millet recipes.", 0.06, 3),
    ("health", "en", "ScienceDaily", "Wearable Gait Sensors Predict Fall Risk in Senior Physiotherapy Patients", "Insole pressure sensors flagged subtle stride asymmetries weeks before clinical balance scores declined.", 0.07, 3),
    ("health", "en", "The Hindu", "Medical Colleges Expand Simulation Labs for Emergency Trauma Triage", "High-fidelity patient mannequins allow junior residents to practice airway management and sepsis protocols before ICU rotations.", 0.05, 3),
    ("health", "en", "BBC", "WHO Publishes Updated Air-Filtration Guidelines for Urban Classrooms", "HEPA and MERV-13 ventilation benchmarks aim to curb seasonal respiratory absenteeism and fine-particulate exposure.", 0.06, 3),
    ("health", "en", "Mint", "Generic Insulin Pen Makers Lower Cold-Chain Thresholds With Heat-Stable Caps", "Phase-change insulated pen caps keep insulin potent at 37 degrees Celsius for up to 28 days without electricity.", 0.07, 3),

    # Education
    ("education", "en", "The Hindu", "Engineering Institutes Replace Rote Coding Labs With Open-Source Audits", "Third-year computer science students now earn lab credits by writing unit tests and fixing documented issues in civic open-source tools.", 0.06, 3),
    ("education", "en", "Indian Express", "State Libraries Open 24-Hour Quiet Reading Rooms for Exam Aspirants", "Municipal public libraries equipped with solar backup lighting and high-speed Wi-Fi recorded 85,000 monthly student check-ins.", 0.05, 3),
    ("education", "en", "Deccan Herald", "Rural High Schools Receive Vernacular Science Experiment Kits", "Hands-on optics, magnetism, and water-testing kits translated into Kannada and Marathi are lifting practical lab participation.", 0.06, 3),
    ("education", "en", "BBC", "Universities Pilot Oral Viva Assessments Alongside Take-Home Essays", "Faculty panels report stronger conceptual understanding when short 10-minute structured oral follow-ups accompany project submissions.", 0.08, 4),
    ("education", "en", "Mint", "Vocational Polytechnics Partner With EV Service Centers for Apprenticeships", "Six-month battery diagnostics and high-voltage safety certifications are placing 88 percent of diploma graduates into regional service bays.", 0.07, 3),
    ("education", "en", "The Hindu", "Bilingual Braille Tactile Graphics Printers Deployed in 60 Resource Centers", "Refreshable tactile embossers allow visually impaired students to inspect geometry diagrams and topographical maps independently.", 0.05, 3),
    ("education", "en", "Indian Express", "National Fellowship Portal Automates Monthly Stipend Disbursements", "Direct benefit transfer tracking cut doctoral research fellowship delays from three months to under five working days.", 0.06, 3),
    ("education", "en", "Deccan Herald", "Teacher Training Colleges Introduce Classroom Acoustics Workshops", "Simple felt panels and seating layouts lowered reverberation noise in crowded primary classrooms, aiding early reading comprehension.", 0.05, 3),

    # Kannada articles (kn) - Regional language requirement
    ("india", "kn", "Prajavani", "ಬೆಂಗಳೂರು ಮೆಟ್ರೋ ಹಳದಿ ಮಾರ್ಗದ ಸಿಗ್ನಲಿಂಗ್ ಪರೀಕ್ಷೆ ಯಶಸ್ವಿ", "ಬೆಂಗಳೂರು ಮೆಟ್ರೋ ರೈಲು ನಿಗಮವು ಆರ್.ವಿ. ರಸ್ತೆಯಿಂದ ಬೊಮ್ಮಸಂದ್ರದವರೆಗಿನ ಹಳದಿ ಮಾರ್ಗದಲ್ಲಿ ಚಾಲಕರಹಿತ ಸಿಗ್ನಲಿಂಗ್ ಪರೀಕ್ಷೆಗಳನ್ನು ಪೂರ್ಣಗೊಳಿಸಿದೆ. ಶೀಘ್ರದಲ್ಲೇ ಸುರಕ್ಷತಾ ಆಯುಕ್ತರ ಪರಿಶೀಲನೆ ನಡೆಯಲಿದೆ.", 0.05, 3),
    ("science", "kn", "Vijayavani", "ಇಸ್ರೋ ಕ್ರಯೋಜೆನಿಕ್ ಎಂಜಿನ್ ಪರೀಕ್ಷೆ ಮಹೇಂದ್ರಗಿರಿಯಲ್ಲಿ ಯಶಸ್ವಿ", "ಭಾರತೀಯ ಬಾಹ್ಯಾಕಾಶ ಸಂಶೋಧನಾ ಸಂಸ್ಥೆ ಇಸ್ರೋ ಮಹೇಂದ್ರಗಿರಿಯಲ್ಲಿ ಸಿಇ-20 ಕ್ರಯೋಜೆನಿಕ್ ಎಂಜಿನ್‌ನ 200 ಸೆಕೆಂಡುಗಳ ಹಾಟ್ ಟೆಸ್ಟ್ ಅನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಪೂರ್ಣಗೊಳಿಸಿದೆ.", 0.04, 2),
    ("technology", "kn", "Prajavani", "ಕನ್ನಡ ದಾಖಲೆಗಳ ಎಐ ಓದುವಿಕೆಗೆ ಹೊಸ ಮುಕ್ತ ಮಾದರಿ ಬಿಡುಗಡೆ", "ಭಾರತೀಯ ಭಾಷೆಗಳ 1.2 ಲಕ್ಷ ಪುಟಗಳ ದಾಖಲೆಗಳನ್ನು ಡಿಜಿಟಲೀಕರಿಸಲು ಮತ್ತು ಸಾರಾಂಶ ನೀಡಲು ಸಂಶೋಧಕರು ಹೊಸ ಮುಕ್ತ ಎಐ ಮಾನದಂಡವನ್ನು ಬಿಡುಗಡೆ ಮಾಡಿದ್ದಾರೆ.", 0.06, 3),
    ("sports", "kn", "Vijayavani", "ಪರ್ತ್ ಟೆಸ್ಟ್: ಭಾರತೀಯ ವೇಗಿಗಳ ಭರ್ಜರಿ ಬೌಲಿಂಗ್‌ನಿಂದ 48 ರನ್ ಜಯ", "ಪರ್ತ್‌ನಲ್ಲಿ ನಡೆದ ಮೊದಲ ಟೆಸ್ಟ್ ಪಂದ್ಯದ ಅಂತಿಮ ದಿನ ಭಾರತದ ವೇಗದ ಬೌಲರ್‌ಗಳು ಏಳು ವಿಕೆಟ್ ಕಬಳಿಸಿ 48 ರನ್‌ಗಳ ರೋಚಕ ಗೆಲುವು ತಂದುಕೊಟ್ಟರು.", 0.07, 2),
    ("education", "kn", "Prajavani", "ಸರ್ಕಾರಿ ಶಾಲಾ ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ರಾಷ್ಟ್ರೀಯ ರೋಬೋಟಿಕ್ಸ್ ಒಲಿಂಪಿಯಾಡ್‌ನಲ್ಲಿ ಪದಕ", "ಪುಣೆಯಲ್ಲಿ ನಡೆದ ರಾಷ್ಟ್ರೀಯ ಶಾಲಾ ರೋಬೋಟಿಕ್ಸ್ ಮತ್ತು ಅಲ್ಗಾರಿದಮ್ಸ್ ಒಲಿಂಪಿಯಾಡ್‌ನಲ್ಲಿ ಕರ್ನಾಟಕದ ಸರ್ಕಾರಿ ವಸತಿ ಶಾಲೆಯ ವಿದ್ಯಾರ್ಥಿಗಳು ಚಿನ್ನದ ಪದಕ ಗೆದ್ದಿದ್ದಾರೆ.", 0.05, 2),
    ("health", "kn", "Vijayavani", "ಗ್ರಾಮೀಣ ಟೆಲಿಮೆಡಿಸಿನ್ ಸೇವೆ 25 ಕೋಟಿ ರೋಗಿಗಳ ಭೇಟಿ ಮೈಲಿಗಲ್ಲು", "ಜಿಲ್ಲಾ ತಜ್ಞ ವೈದ್ಯರನ್ನು ಪ್ರಾಥಮಿಕ ಆರೋಗ್ಯ ಕೇಂದ್ರಗಳಿಗೆ ಸಂಪರ್ಕಿಸುವ ಟೆಲಿಮೆಡಿಸಿನ್ ಜಾಲವು 25 ಕೋಟಿ ಸಮಾಲೋಚನೆಗಳನ್ನು ಪೂರ್ಣಗೊಳಿಸಿದೆ.", 0.05, 2),
    ("business", "kn", "Prajavani", "ಸಣ್ಣ ವ್ಯಾಪಾರಿಗಳಿಗೆ ಯುಪಿಐ ಲೈಟ್ ಆಫ್‌ಲೈನ್ ಮಿತಿ 1,000 ರೂ.ಗೆ ಏರಿಕೆ", "ಭಾರತೀಯ ರಿಸರ್ವ್ ಬ್ಯಾಂಕ್ ಪಿನ್ ರಹಿತ ಯುಪಿಐ ಲೈಟ್ ವಹಿವಾಟಿನ ಮಿತಿಯನ್ನು 1,000 ರೂಪಾಯಿಗೆ ಹೆಚ್ಚಿಸಿರುವುದು ಸಣ್ಣ ವ್ಯಾಪಾರಿಗಳಿಗೆ ಅನುಕೂಲವಾಗಿದೆ.", 0.06, 2),
    ("entertainment", "kn", "Vijayavani", "ಗೋವಾ ಚಲನಚಿತ್ರೋತ್ಸವದಲ್ಲಿ ಕನ್ನಡದ ಸ್ವತಂತ್ರ ಸಿನಿಮಾಗಳಿಗೆ ಮೆಚ್ಚುಗೆ", "ಪ್ರಾದೇಶಿಕ ಸಿನಿಮಾ ವಿಭಾಗದಲ್ಲಿ ಪ್ರದರ್ಶನಗೊಂಡ ಕನ್ನಡದ ನವ ನಿರ್ದೇಶಕರ ಚಿತ್ರಗಳು ವಿಮರ್ಶಕರಿಂದ ಉತ್ತಮ ಪ್ರತಿಕ್ರಿಯೆ ಪಡೆದಿವೆ.", 0.06, 2),

    # Hindi articles (hi) - Regional language requirement
    ("science", "hi", "Dainik Bhaskar", "इसरो ने महेंद्रगिरि में क्रायोजेनिक इंजन का 200 सेकंड का सफल परीक्षण किया", "भारतीय अंतरिक्ष अनुसंधान संगठन इसरो ने महेंद्रगिरि केंद्र में सीई-20 क्रायोजेनिक इंजन का 200 सेकंड का हॉट टेस्ट सफलतापूर्वक पूरा किया। इससे आगामी गगनयान और एलवीएम3 मिशनों को बल मिलेगा।", 0.05, 3),
    ("business", "hi", "Amar Ujala", "आरबीआई ने यूपीआई लाइट की ऑफलाइन लेनदेन सीमा बढ़ाकर 1,000 रुपये की", "भारतीय रिजर्व बैंक ने बिना इंटरनेट और पिन के छोटे डिजिटल भुगतान को बढ़ावा देने के लिए यूपीआई लाइट की प्रति लेनदेन सीमा बढ़ाकर 1,000 रुपये और वॉलेट सीमा 5,000 रुपये कर दी है।", 0.06, 2),
    ("technology", "hi", "Dainik Bhaskar", "हिंदी और कन्नड़ दस्तावेजों के लिए नया ओपन-सोर्स एआई बेंचमार्क जारी", "शोधकर्ताओं ने 14 भारतीय लिपियों के 1.2 लाख पृष्ठों के सटीक ओसीआर और सारांश मूल्यांकन के लिए एक नया ओपन-सोर्स एआई डेटासेट जारी किया है।", 0.05, 3),
    ("sports", "hi", "Amar Ujala", "पर्थ टेस्ट में भारतीय तेज गेंदबाजों का कमाल, 48 रन से ऐतिहासिक जीत", "पर्थ में खेले गए पहले टेस्ट मैच के अंतिम सत्र में भारतीय तेज गेंदबाजों ने सात विकेट झटककर टीम इंडिया को 48 रनों से शानदार जीत दिलाई।", 0.07, 2),
    ("health", "hi", "Dainik Bhaskar", "डेंगू वैक्सीन के तीसरे चरण के ट्रायल में 81 प्रतिशत प्रभावशीलता दर्ज", "देश के 18 चिकित्सा केंद्रों पर 10,000 प्रतिभागियों पर किए गए डेंगू टीके के तीसरे चरण के नैदानिक परीक्षण में 81 प्रतिशत सुरक्षा दर्ज की गई है।", 0.05, 3),
    ("education", "hi", "Amar Ujala", "यूजीसी ने 420 विश्वविद्यालयों में एकेडमिक बैंक ऑफ क्रेडिट पोर्टल शुरू किया", "विश्वविद्यालय अनुदान आयोग के नए पोर्टल से छात्र अपने कौशल पाठ्यक्रम और इंटर्नशिप क्रेडिट को देशभर के 420 विश्वविद्यालयों में स्थानांतरित कर सकेंगे।", 0.05, 2),
    ("india", "hi", "Dainik Bhaskar", "वेस्टर्न डेडिकेटेड फ्रेट कॉरिडोर से माल ढुलाई का समय घटकर 26 घंटे हुआ", "गुजरात के बंदरगाहों से राष्ट्रीय राजधानी क्षेत्र तक डबल-स्टैक कंटेनर ट्रेनों की औसत यात्रा अवधि 68 घंटे से घटकर 26 घंटे रह गई है।", 0.06, 2),
    ("world", "hi", "Amar Ujala", "जलवायु सम्मेलन में 140 देशों ने स्वच्छ ग्रिड और बैटरी फंड पर सहमति जताई", "वैश्विक जलवायु वार्ता में 140 देशों के प्रतिनिधियों ने विकासशील देशों में सौर और पवन ऊर्जा भंडारण ग्रिड विकसित करने के लिए नए वित्तपोषण ढांचे को मंजूरी दी।", 0.05, 3),

    # Intentional Clickbait Examples (for clickbait detector testing & UI flags, score > 0.6)
    ("technology", "en", "ViralByte", "YOU WON'T BELIEVE What This Secret Smartphone Setting Does To Your Battery!!", "A routine software update adjusts background refresh intervals by about four percent over an eight-hour standby window according to standard developer documentation.", 0.88, 2),
    ("health", "en", "ClickDaily", "SHOCKING!! Doctors HATE This One Weird Morning Water Habit?!", "Staying adequately hydrated after waking up supports normal kidney function and mild alertness, nutritionists explain in standard dietary guidelines.", 0.91, 2),
    ("business", "en", "BuzzFinance", "7 SECRETS Banks Don't Want You To Know About Savings Accounts!!", "Comparing annual percentage yields and checking whether monthly maintenance fees apply are basic steps when opening a deposit account.", 0.84, 2),
    ("entertainment", "en", "TrendPulse", "JAW-DROPPING! What Happened Next On Set Broke The Internet!!", "A scheduled behind-the-scenes lighting rehearsal lasted fifteen minutes longer than planned before the cast resumed normal filming.", 0.89, 2),
    ("science", "en", "AstroBuzz", "UNBELIEVABLE Space Discovery That Changes EVERYTHING Forever?!", "Astronomers cataloged a faint asteroid passing safely four million kilometers beyond lunar orbit during routine sky surveys.", 0.86, 2),
    ("sports", "en", "GoalHype", "INSANE Locker Room Move Nobody Expected Before Kickoff!!", "The coaching staff rotated two midfield substitutes during pre-match warmups to manage minor hamstring soreness.", 0.82, 2),
    ("technology", "en", "TechHype", "EXPOSED: The MIND-BLOWING Reason Your Wi-Fi Slows Down At Night!!", "Higher household video streaming traffic during evening hours shares available router bandwidth across connected devices.", 0.85, 2),
    ("india", "en", "CityScoop", "You Need To See This SHOCKING Traffic Rule Change Right Now!!", "Municipal traffic police installed clearer lane markings and reflective signage along two ring road junctions ahead of roadworks.", 0.80, 2),
    ("education", "en", "CampusBuzz", "THIS ONE TRICK Top Students Use That Universities Never Mention?!", "Reviewing lecture notes within twenty-four hours and practicing spaced recall improves long-term exam retention.", 0.83, 2),
    ("world", "en", "GlobeFlash", "BOMBSHELL Flight Delay Secret Airlines Are Hiding From Passengers!!", "Winter runway de-icing procedures and air traffic spacing regularly add ten to twenty minutes to departure schedules.", 0.87, 2),

    # Additional clean factual articles to reach exactly 200 articles
    ("technology", "en", "TechCrunch", "Compiler Team Cuts Rust Incremental Build Times by 28 Percent", "Parallelized metadata serialization in the latest compiler nightly reduced incremental rebuild latency on large workspace crates.", 0.05, 3),
    ("technology", "en", "Ars Technica", "Home Energy Monitors Adopt Open Matter 1.4 Telemetry Standard", "Smart circuit breakers and rooftop solar inverters can now exchange wattage telemetry locally without cloud vendor lock-in.", 0.06, 3),
    ("business", "en", "Mint", "Warehousing Hubs Shift Toward Automated Pallet Shuttles", "E-commerce fulfillment centers in Bhiwandi and Hosur are installing high-density shuttle racks to speed order picking.", 0.06, 3),
    ("business", "en", "Economic Times", "Export Credit Agency Expands Micro-Insurance for Handicraft Artisans", "Small craft cooperatives shipping orders abroad can now purchase per-shipment transit insurance directly through postal export centers.", 0.05, 3),
    ("science", "en", "ScienceDaily", "Coral Nursery Project Reports 70 Percent Survival Rate After Heatwave", "Heat-tolerant Acropora fragments cultivated in shaded mid-water racks withstood peak sea-surface anomalies in Gulf of Mannar trials.", 0.05, 3),
    ("science", "en", "Nature Briefing", "Acoustic Monitoring Tracks Recovery of Mangrove Bird Populations", "Automated solar audio recorders logged a steady rebound in resident kingfisher and warbler calls across restored coastal mudflats.", 0.04, 3),
    ("india", "en", "The Hindu", "Public Transit Agencies Unify Smart Card Transfers Across Metro and Bus", "Commuters can now tap a single NCMC debit card across suburban buses, metro gates, and feeder shuttles with capped daily fares.", 0.06, 3),
    ("india", "en", "Deccan Herald", "Municipal Composting Yards Convert Market Waste Into City Park Mulch", "Segregated vegetable market organics are processed in aerated windrows within 21 days for urban tree-planting drives.", 0.05, 3),
    ("world", "en", "Reuters", "International Aviation Body Approves Sustainable Fuel Book-and-Claim Registry", "Airlines and corporate freight buyers can now verify lifecycle emissions reductions from certified bio-jet fuel batches.", 0.07, 4),
    ("world", "en", "BBC", "Alpine Glaciologists Install Solar Radar Stations on High Ridges", "Continuous ground-penetrating radar measurements will feed real-time meltwater models for downstream hydropower reservoirs.", 0.05, 3),
    ("sports", "en", "The Hindu", "Table Tennis Federation Launches Regional Biomechanics Camps", "Junior national paddlers are analyzing footwork recovery speed and spin-return angles using high-frame-rate court cameras.", 0.06, 3),
    ("sports", "en", "ESPNcricinfo", "Groundskeepers Adopt Subsurface Drainage Sensors Ahead of Wet Season", "Moisture probes beneath cricket outfields allow stadium crews to target aeration and super-sopper passes efficiently.", 0.05, 3),
    ("health", "en", "The Hindu", "Community Optometry Vans Screen 40,000 Rural Drivers for Night Vision", "Mobile eye clinics distributed prescription anti-glare spectacles to highway truck and bus drivers at rest stops.", 0.05, 3),
    ("health", "en", "Indian Express", "Biomedical Engineers Design Low-Cost Neonatal Phototherapy Cradles", "Narrow-band LED phototherapy units built with modular power supplies simplify newborn jaundice care in primary clinics.", 0.04, 3),
    ("education", "en", "The Hindu", "Open Textbook Consortium Publishes Free Calculus Texts in Six Languages", "University mathematics professors released peer-reviewed problem banks under Creative Commons licenses for undergraduate engineering classes.", 0.05, 3),
    ("education", "en", "Deccan Herald", "School Bus Fleet Pilots Blind-Spot Radar and Parent Arrival Alerts", "Safety upgrades across 300 school buses pair ultrasonic curb sensors with encrypted stop-notification pings.", 0.06, 2),
    ("entertainment", "en", "The Hindu", "Sound Designers Recreate Acoustic Heritage of Historic Stepwells", "Architectural acousticians recorded impulse responses inside centuries-old stone monuments for museum spatial exhibitions.", 0.05, 3),
    ("entertainment", "en", "BBC", "Independent Comic Artists Form Print Co-Op for Graphic Non-Fiction", "Illustrators pooled offset printing runs to distribute science and history graphic novels across neighborhood libraries.", 0.06, 3),
    ("technology", "en", "Mint", "Open-Source Vector Databases Optimize Disk-Backed HNSW Indexes", "Memory-mapped graph indexes allow billion-scale semantic search nodes to run on modest RAM budgets.", 0.05, 4),
    ("business", "en", "Business Standard", "Specialty Coffee Growers Adopt Direct Traceability QR Lots", "Chikmagalur estates are tagging washed arabica micro-lots with roast-curve metadata for specialty roasters in Europe and Japan.", 0.06, 3),
    ("science", "en", "ScienceDaily", "Atmospheric Chemists Measure Urban Ozone Drop After Electric Bus Rollout", "Roadside air-quality monitors recorded measurable reductions in nitrogen oxide peaks along bus rapid transit corridors.", 0.05, 3),
    ("india", "en", "Indian Express", "State Archives Digitize Century-Old Monsoon Rainfall Ledgers", "Volunteer citizen transcribers helped convert handwritten taluk rain-gauge registers from 1890 to 1960 into open CSV tables.", 0.05, 3),
    ("world", "en", "Reuters", "Port of Rotterdam Commissions Shore-Power Substations for Container Ships", "Docked cargo vessels can now plug into land-based renewable electricity instead of idling auxiliary diesel generators.", 0.06, 3),
    ("sports", "en", "BBC", "Rowing Regatta Organizers Adopt GPS Buoy Courses to Cut Boat Wake", "Automated virtual lane markers replaced anchored concrete weights, protecting shallow lakebed seagrass beds.", 0.05, 3),
    ("health", "en", "ScienceDaily", "Physiotherapists Validate Tele-Rehab Protocols After Knee Surgery", "Structured video-guided mobility programs produced comparable six-month range-of-motion recovery to in-clinic sessions.", 0.06, 3),
    ("education", "en", "Indian Express", "Public Museums Offer Weekend Science Illustration Workshops for Teens", "Botanical and astronomical sketching classes bridge art education and scientific observation for high schoolers.", 0.05, 2),
    ("entertainment", "en", "Deccan Herald", "Independent Music Venues Standardize Sound-Level Monitoring for Hearing Safety", "Acoustic limiters and free hi-fi earplug dispensers are becoming standard across small live-music clubs.", 0.05, 3),
    ("technology", "en", "Ars Technica", "RISC-V Laptop Firmware Achieves Upstream UEFI Boot Support", "Developers demonstrated mainline firmware booting standard Linux installers cleanly on open-ISA developer notebooks.", 0.06, 4),
    ("business", "en", "Mint", "Cooperative Millet Bakeries Scale Vacuum-Packed Whole-Grain Snacks", "Women-led rural processing units secured retail shelf space across airport kiosks and railway stations.", 0.05, 3),
    ("science", "en", "The Hindu", "Astrophysics Students Build Low-Cost Interferometer for Solar Radio Bursts", "Dual dish antennas on a university rooftop successfully tracked Type II solar radio bursts during active sunspot cycles.", 0.06, 3),
    ("india", "en", "Deccan Herald", "Civic Hackers Publish Open GTFS Feeds for 14 Tier-2 Bus Networks", "Standardized transit timetable files now allow open-source map apps to display live bus routing in smaller cities.", 0.05, 3),
    ("world", "en", "BBC", "Mediterranean Marine Reserves Report Biomass Rebound in No-Take Zones", "Ten-year underwater visual censuses confirmed larger grouper and sea-bream populations spilling over into neighboring artisanal fisheries.", 0.06, 3),
    ("sports", "en", "Indian Express", "National Cycling Velodrome Installs Electronic Lap-Split Transponders", "Track sprinters receive millisecond sector splits immediately after flying 200-meter qualifying runs.", 0.05, 2),
    ("health", "en", "Deccan Herald", "Urban Parks Add Shaded Walking Loops Marked With Distance Signage", "Simple 400-meter loop markers and hydration fountains encouraged higher daily step counts among older residents.", 0.05, 2),
    ("education", "en", "Mint", "Architecture Schools Introduce Bamboo and Compressed-Earth Structural Labs", "Civil and architecture undergraduates load-tested low-carbon regional masonry blocks in campus structural frames.", 0.06, 3),
    ("entertainment", "en", "The Hindu", "Documentary Filmmakers Launch Shared Archival Footage Licensing Co-Op", "Independent documentarians created a transparent micro-licensing pool for regional historical newsreel clips.", 0.05, 3),
]


def _make_short_bullets(body: str) -> str:
    parts = [p.strip() for p in body.split(". ") if p.strip()]
    bullets = [f"- {p.rstrip('.')}." for p in parts[:3]]
    while len(bullets) < 3:
        bullets.append(f"- {parts[0].rstrip('.')}.")
    return "\n".join(bullets)


def build_datasets() -> None:
    articles: list[dict] = []
    idx = 1

    # 1. Expand multi-source clusters
    for clu in MULTI_SOURCE_CLUSTERS:
        core_title, core_body = clu["core"]
        for offset_min, (source, variant_title) in enumerate(clu["variants"]):
            pub = BASE_TIME - timedelta(hours=clu["hours_ago"], minutes=offset_min * 25)
            body = f"{core_body} ({source} dispatch)."
            articles.append(
                {
                    "id": f"art_{idx:03d}",
                    "title": variant_title,
                    "url": f"https://news.example.org/{clu['topic']}/art-{idx:03d}",
                    "source": source,
                    "published_at": pub.isoformat().replace("+00:00", "Z"),
                    "topic": clu["topic"],
                    "language": clu["language"],
                    "summary_short": _make_short_bullets(body),
                    "summary_long": body,
                    "body": body,
                    "cluster_id": clu["cluster_id"],
                    "clickbait_score": 0.07,
                    "reading_time_min": 3,
                    "why_reason": "",
                }
            )
            idx += 1

    # 2. Add standalone articles (English, Kannada, Hindi, Clickbait)
    for k, (topic, lang, source, title, body, cb_score, rtime) in enumerate(STANDALONE_ARTICLES):
        pub = BASE_TIME - timedelta(hours=1 + (k % 46), minutes=(k * 13) % 60)
        articles.append(
            {
                "id": f"art_{idx:03d}",
                "title": title,
                "url": f"https://news.example.org/{topic}/art-{idx:03d}",
                "source": source,
                "published_at": pub.isoformat().replace("+00:00", "Z"),
                "topic": topic,
                "language": lang,
                "summary_short": _make_short_bullets(body),
                "summary_long": body,
                "body": body,
                "cluster_id": f"clu_single_{idx:03d}",
                "clickbait_score": cb_score,
                "reading_time_min": rtime,
                "why_reason": "",
            }
        )
        idx += 1

    # Ensure exact count of 200 articles
    extra_templates = [
        ("technology", "en", "TechCrunch", "Browser DevTools Add Automated Accessibility Contrast Audits", "Built-in developer panels now flag low-contrast interactive buttons and missing keyboard focus rings during local page inspections."),
        ("science", "en", "ScienceDaily", "Biochar Soil Trials Improve Moisture Retention in Rainfed Cotton Plots", "Three-season agronomic plots demonstrated higher root-zone moisture persistence during mid-monsoon dry spells."),
        ("business", "en", "Mint", "Regional Cold-Storage Co-Ops Cut Tomato Price Volatility", "Farmer producer organizations operating solar chillers staggered market arrivals across four harvest weeks."),
        ("india", "en", "The Hindu", "Urban Forestry Drives Plant Native Canopy Corridors Along Ring Roads", "Botanists selected drought-resilient native shade species rather than fast-growing exotics for urban avenue planting."),
        ("world", "en", "Reuters", "Global Rail Freight Consortium Standardizes Digital Customs Manifests", "Electronic consignment notes trimmed border checkpoint dwell times along Eurasian rail corridors."),
        ("sports", "en", "The Hindu", "University Athletics Boards Introduce Concussion Return-to-Play Protocols", "Mandatory 48-hour medical clearance checks now govern collegiate rugby, football, and boxing tournaments."),
        ("health", "en", "BBC", "Digital Hearing Aid Calibration Apps Expand Access in Community Clinics", "Audiologists trained primary nurses to fit programmable hearing aids using validated tablet tone-sweep tests."),
        ("education", "en", "Deccan Herald", "Public Makerspaces Host Weekend Repair Cafes for Engineering Freshers", "Students learn soldering, multimeter diagnostics, and appliance repair alongside retired workshop instructors."),
        ("entertainment", "en", "Indian Express", "Regional Folk Music Archives Release Open Stem Recordings for Remixers", "Cultural trusts published clean multi-track folk percussion stems under non-commercial attribution licenses."),
        ("technology", "en", "Ars Technica", "WebAssembly Component Model Simplifies Polyglot Plugin Sandboxes", "Applications can now embed isolated Python, Rust, and C++ filters inside a unified capability-safe runtime."),
    ]
    extra_i = 0
    while len(articles) < 200:
        topic, lang, source, title, body = extra_templates[extra_i % len(extra_templates)]
        pub = BASE_TIME - timedelta(hours=3 + extra_i * 2)
        articles.append(
            {
                "id": f"art_{idx:03d}",
                "title": title,
                "url": f"https://news.example.org/{topic}/art-{idx:03d}",
                "source": source,
                "published_at": pub.isoformat().replace("+00:00", "Z"),
                "topic": topic,
                "language": lang,
                "summary_short": _make_short_bullets(body),
                "summary_long": body,
                "body": body,
                "cluster_id": f"clu_single_{idx:03d}",
                "clickbait_score": 0.06,
                "reading_time_min": 3,
                "why_reason": "",
            }
        )
        idx += 1
        extra_i += 1

    articles = articles[:200]
    (DATA_DIR / "sample.json").write_text(
        json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3. 50 hand-labeled classification evaluation examples (balanced across all 9 topics + kn/hi)
    eval_cls = []
    by_topic: dict[str, list[dict]] = {}
    for a in articles:
        if a["clickbait_score"] < 0.5:
            by_topic.setdefault(a["topic"], []).append(a)
    per_topic = 5
    for topic, items in sorted(by_topic.items()):
        for item in items[:per_topic]:
            eval_cls.append(
                {
                    "id": item["id"],
                    "text": f"{item['title']}. {item['body']}",
                    "language": item["language"],
                    "true_topic": item["topic"],
                }
            )
    # Top up to 50 with remaining non-clickbait items
    used_ids = {x["id"] for x in eval_cls}
    for a in articles:
        if len(eval_cls) >= 50:
            break
        if a["id"] not in used_ids and a["clickbait_score"] < 0.5:
            eval_cls.append(
                {
                    "id": a["id"],
                    "text": f"{a['title']}. {a['body']}",
                    "language": a["language"],
                    "true_topic": a["topic"],
                }
            )
            used_ids.add(a["id"])

    (DATA_DIR / "eval_classification_50.json").write_text(
        json.dumps(eval_cls[:50], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. 30 hand-labeled clickbait evaluation examples (10 clickbait + 20 non-clickbait)
    cb_pos = [
        {"id": a["id"], "title": a["title"], "body": a["body"], "is_clickbait": True}
        for a in articles
        if a["clickbait_score"] >= 0.6
    ][:10]
    cb_neg = [
        {"id": a["id"], "title": a["title"], "body": a["body"], "is_clickbait": False}
        for a in articles
        if a["clickbait_score"] < 0.3 and a["language"] == "en"
    ][:20]
    (DATA_DIR / "eval_clickbait_30.json").write_text(
        json.dumps(cb_pos + cb_neg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Generated {len(articles)} sample articles, "
        f"{len(eval_cls[:50])} classification eval items, and "
        f"{len(cb_pos + cb_neg)} clickbait eval items."
    )


if __name__ == "__main__":
    build_datasets()
