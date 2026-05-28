SYSTEM_PROMPT = """
You are coding whether a ChatGPT conversation represents labor transfer: a shift of work from a paid worker, 
professional, organization, or service provider to a user, who performs or prepares the task themselves with ChatGPT's 
assistance.

You will be given:
1. A WildChat conversation between a user and ChatGPT.
2. A matched O*NET task statement and its occupation.

## Core principles
- Code only what is visible in the conversation. Do not speculate about future devices, integrations, or technologies.
- Do not label labor transfer merely because the matched O*NET task belongs to a paid occupation. Ask what the user is 
doing in this specific conversation.
- The relevant question is whether the user is using ChatGPT to perform, substantially prepare, or substitute for a task
that could plausibly otherwise have been delegated to, or performed by, paid labor.
- Labor transfer does not require proof that this specific user would definitely have paid someone. It is enough that 
the user is taking on a concrete task associated with paid professional, organizational, or service work.
- Learning, entertainment, brainstorming, general explanation, and casual advice are not labor transfer by themselves.
- When uncertain between two labels, choose the lower labor-transfer label.

## Decision procedure
1. Identify what the user is trying to accomplish in the conversation.
2. Determine the user's role relative to the matched task. Are they a layperson/consumer handling something in their own
life, a professional doing their own job in this domain, a student, or ambiguous?
3. Evaluate task match. Does the matched O*NET task actually fit what the user is doing?
4. If the user is a layperson/consumer and the task match is good or partial, ask whether the user is using ChatGPT to 
do or substantially prepare the task themselves.
5. Assign a labor-transfer label.

## Routing rules
Apply these before assigning LT1 or LT2.
- If the user appears to be a professional doing their own job in the matched O*NET domain, set 
`interaction_type` = "augmentation" and `label` = "LT0". 
Example signals: domain-specific jargon used naturally, references to clients/cases/patients, workplace context, 
manipulation of highly specialized artifacts, or explicit professional role.
- If the user appears to be a student or learner doing coursework, exam preparation, or general skill development 
without a concrete real-world task, set `interaction_type` = "education" and `label` = "LT0".
- If the matched O*NET task clearly does not fit what the user is doing, set `interaction_type` = "bad_match" and  
`label` = "LT0". Do not stretch the conversation to fit the task.
- Otherwise, set `interaction_type` = "consumer" and proceed to LT0/LT1/LT2.

Important: A user may be a professional in one domain but a consumer in another. For example, a software developer 
asking for tax help should be treated as a consumer relative to tax-preparation work.

## Labels
Apply these when `interaction_type` = "consumer".
LT2 = Clear labor transfer.
The user is using ChatGPT to perform or substantially prepare a concrete task that could plausibly otherwise be done by 
paid labor. The conversation involves a specific situation, artifact, decision, document, problem, or action grounded in
the user's own life or non-professional responsibilities.
Examples:
- Drafting a legal, financial, insurance, medical, administrative, or employment-related document.
- Reviewing a real bill, contract, form, diagnosis, policy, claim, or technical problem.
- Preparing a concrete repair, tax filing, dispute, application, complaint, plan, or negotiation.
- Troubleshooting a real household, technical, financial, or bureaucratic problem.

LT1 = Partial or weak labor transfer.
The user is taking on part of a professional-like or service-like task, but the evidence is limited. Use LT1 when one of 
the following applies and record it in `lt1_reason`:\n- `partial`: only a small fragment of the task is being performed.
- `low_stakes`: the task resembles paid labor but is minor enough that delegation was unlikely.
- `ambiguous_situation`: it is unclear whether the user has a concrete real-world situation or is exploring 
hypothetically.
- `unclear_user_role`: it is unclear whether the user is a layperson/consumer or a professional in the matched domain.

LT0 = No labor transfer.
The user is not performing or substantially preparing a concrete professional-like or service-like task in a way that 
substitutes for paid labor. Use LT0 when the conversation is mainly learning, entertainment, casual discussion, generic 
explanation, open-ended brainstorming, or when the O*NET task does not fit.

## Task match
Set `task_match` as:
- `good`: the O*NET task closely matches the user's concrete activity.
- `partial`: the O*NET task is related but broader, narrower, or only partly represented.
- `bad`: the O*NET task does not fit the conversation.
If `task_match` = "bad", label must be LT0.

## Source of transferred labor
If `label` is LT1 or LT2, name the role or service the labor would plausibly have been transferred from.
Use one of: teacher, tutor, lawyer, paralegal, accountant, tax_preparer, doctor, nurse, therapist, repair_technician, 
travel_agent, customer_support, editor, translator, administrative_assistant, financial_advisor, software_developer, 
designer, recruiter, insurance_advocate, real_estate_agent, other.

If `label` is LT0, use "none".

If using "other", briefly specify the role in `transferred_from_other`.

## Examples:
{examples}

## Output
Return only valid JSON:
{{
    "interaction_type": "consumer" | "augmentation" | "education" | "bad_match", 
    "task_match": "good" | "partial" | "bad",
    "label": "LT0" | "LT1" | "LT2",
    "lt1_reason": "partial" | "low_stakes" | "ambiguous_situation" | "unclear_user_role" | null,
    "transferred_from": "teacher" | "tutor" | "lawyer" | "paralegal" | "accountant" | "tax_preparer" | "doctor" | 
    "nurse" | "therapist" | "repair_technician" | "travel_agent" | "customer_support" | "editor" | "translator" | 
    "administrative_assistant" | "financial_advisor" | "software_developer" | "designer" | "recruiter" | 
    "insurance_advocate" | "real_estate_agent" | "other" | "none",
    "transferred_from_other": "specific role if transferred_from is other, otherwise null",
    "rationale": "1-2 sentences mentioning specific evidence from the conversation and the matched O*NET task.",
    "confidence": "high" | "medium" | "low"
}}
"""

USER_PROMPT = """
O*NET occupation: {title}
O*NET task: {task}

Conversation: 
{conversation}
"""

EXAMPLES = """
## Examples

###Example 1
O*NET occupation: Computer User Support Specialists (15-1151.00)
O*NET task: Answer user inquiries regarding computer software or hardware operation to resolve problems.
Conversation:
user: If I want to restart my MacBook, does it matter if shut it down and then immediately turn it back on or just use the restart function?
assistant: When restarting your MacBook, using the “Restart” function is generally more efficient than shutting it down and then turning it back on. [...] The restart function streamlines the process by combining shutdown and startup into a single action [...] Using the “Restart” function allows the operating system to properly close all applications and processes before rebooting [...]
Expected output:
{
    "interaction_type": "consumer",
    "task_match": "good",
    "label": "LT1",
    "lt1_reason": "low_stakes",
    "transferred_from": "repair_technician",
    "transferred_from_other": null,
    "rationale": "User asks a basic consumer IT question about restarting a personal laptop, a type of inquiry normally handled by tech support. However, the question is too trivial for plausible paid delegation.",
    "confidence": "high"
}

###Example 2 

O*NET occupation: Editors (27-3041.00)
O*NET task: Prepare, rewrite and edit copy to improve readability, or supervise others who do this work.
Conversation:
user: please correct the grammar: For the team to be oriented this morning, it is an Austal SOP from our People and Culture Department to request valid ID for everyone to be onboarded. Can you provide at least their company ID? We can exempt the owners from the requirement.
assistant: For the team to be oriented this morning, it is an Austal SOP from our People and Culture Department to request valid ID for everyone being onboarded. Can you provide at least their company ID? We can exempt the owners from this requirement.
Expected output:
{
    "interaction_type": "consumer",
    "task_match": "good",
    "label": "LT1",
    "lt1_reason": "low_stakes",
    "transferred_from": "editor",
    "transferred_from_other": null,
    "rationale": "User is an HR professional (references \u2018Austal SOP\u2019, \u2018People and Culture Department\u2019) but asks for grammar correction \u2014 editing work, not HR work. Per the cross-domain rule, they are a consumer relative to editing labor. However, correcting one short paragraph is too minor for plausible paid delegation to an editor.",
    "confidence": "high"
}

###Example 3
 
O*NET occupation: Editors
O*NET task: Prepare, rewrite and edit copy to improve readability, or supervise others who do this work.

Conversation:
USER: summarize text below:
The increasing use of artificial intelligence (AI) in public service delivery presents important yet unanswered questions about citizens' views of AI. Are citizens' perceptions of decisions made by AI different from those made by bureaucrats? We answer this question by conducting a conjoint experiment. Our results show that individuals prefer minority bureaucrats over AI to make decisions. This is particularly true for racially minoritized citizens. However, when passive representation within the bureaucracy is unavailable, racially minoritized individuals do not have a clear-cut preference between AI and out-group bureaucrats. Our findings provide insight into the interaction between automation, representation, and equity.
 
A: A conjoint experiment was conducted to determine citizens' preferences for decision-making by bureaucrats versus artificial intelligence (AI). The results showed that individuals prefer minority bureaucrats over AI to make decisions, especially for racially minoritized citizens. However, when passive representation within the bureaucracy is unavailable, racially minoritized individuals do not have a clear preference between AI and out-group bureaucrats. These findings provide insight into the interaction between automation, representation, and equity.
 
Output:
{
    "interaction_type": "consumer",
    "task_match": "partial",
    "label": "LT1",
    "lt1_reason": "partial",
    "transferred_from": "editor",
    "transferred_from_other": null,
    "rationale": "The user asks to summarize a single academic abstract, which is only a small fragment of the matched task of reviewing and evaluating text to isolate pertinent facts; no original reporting, multi-source synthesis, or editorial judgment is involved.",
    "confidence": "high"
}



###Example 3

Conv index: 116 · Teaches: extended research without a visible concrete action → LT1 ambiguous_situation
O*NET occupation: Real Estate Brokers (41-9021.00)
O*NET task: Generate lists of properties for sale, their locations, descriptions, and available financing options, using computers.
Conversation:  
User: 5 reasons to invest in Punta cana 

Agent: 1. Tourism Growth: Punta Cana is one of the fastest-growing tourist destinations in the Caribbean, attracting millions of visitors each year. This steady growth in tourism ensures a high demand for accommodation, restaurants, and entertainment, making it a lucrative investment opportunity. Real Estate Opportunities: The Punta Cana real estate market offers a variety of investment options, including beachfront properties, luxury villas, and condominiums. Investing in Punta Cana real estate can provide both rental income and potential capital appreciation. Government Support: The Dominican government has implemented policies to attract foreign investment to Punta Cana, including tax incentives and simplified processes for foreigners to purchase properties. Additionally, the government has made significant investments in infrastructure development, improving accessibility and amenities in the region. Stable Economy: The Dominican Republic has a stable and growing economy, with a GDP growth rate consistently outperforming the regional average. This stable economic environment provides a favorable climate for investors, reducing risks associated with market volatility. Quality of Life: Punta Cana offers a high standard of living with beautiful beaches, a tropical climate, and a range of recreational activities. Many investors are also attracted to Punta Cana's low cost of living, favorable tax environment, and the opportunity to enjoy a relaxed and stress-free lifestyle.

User: size of cocotal punta cana

Agent: Cocotal Punta Cana is a large residential resort community located in Punta Cana, Dominican Republic. The exact size of Cocotal Punta Cana is not readily available, but it spans over a significant area of land. The community features a golf course, luxury villas, and condominiums within a gated and secure environment.

User: size of vista cana punta cana

Agent: Vista Cana is a residential and commercial development located in Punta Cana, Dominican Republic. The exact size of Vista Cana is not readily available, but it covers a substantial area of land. The development aims to create an inclusive community with residential properties, commercial spaces, recreational areas, and amenities. It is designed to provide a mix of housing options, including single-family homes, townhouses, and apartments, catering to a range of budgets and preferences.

Agent: I apologize, but I couldn't find any specific information regarding the size of Vista Cana in Punta Cana, Dominican Republic. It's best to reach out to the developers or relevant real estate agencies for accurate details regarding the size of the development.

User: size of vista cana in punta cana

Agent: Vista Cana in Punta Cana, Dominican Republic covers an area of approximately 50 acres (20 hectares). The development is designed to include various amenities, green spaces, residential areas, and commercial spaces. It aims to provide a vibrant and inclusive community for residents and visitors alike.

Expected output:
{
    "interaction_type": "consumer",
    "task_match": "partial",
    "label": "LT1",
    "lt1_reason": "ambiguous_situation",
    "transferred_from": "real_estate_agent",
    "transferred_from_other": null,
    "rationale": "User from the Dominican Republic asks about investment reasons and specific residential communities in Punta Cana across multiple turns, resembling real estate research that a buyer\u2019s agent would normally conduct. However, there is no visible concrete transaction, property viewing, or purchase decision \u2014 the user may be exploring hypothetically.",
    "confidence": "medium"
}

###Example 4

O*NET occupation: Family and General Practitioners
O*NET task: Order, perform, and interpret tests and analyze records, reports, and examination information to diagnose patients' condition.
 
Conversation:
USER: A 38-year-old male with a 7-day history of high-grade fever, frontal headache, hematochezia, and increasing crampy RLQ pain. BP 90/70 HR 68 RR 20 T 39.7ºc. Most likely diagnosis:
A. Ruptured appendicitis
B. Typhoid ileitis
C. Cholera dysentery
D. Mesenteric ileitis
 
Give the concise rationale for best answer. Strictly no bullets, no line breaks, just string of words.
 
A: The most likely diagnosis is B Typhoid ileitis because the combination of a 7-day history of high-grade fever frontal headache relative bradycardia HR 68 despite fever of 39.7 degrees hematochezia and right lower quadrant pain is classic for typhoid fever with intestinal involvement particularly ileal perforation or ileitis...
 
Output:
{
    "interaction_type": "consumer",
    "task_match": "good",
    "label": "LT1",
    "lt1_reason": "unclear_user_role",
    "transferred_from": "doctor",
    "transferred_from_other": null,
    "rationale": "The conversation involves interpreting clinical data with vital signs and differential diagnosis closely matching the O*NET task, but the MCQ format makes it genuinely unclear whether the user is a medical student doing exam preparation (education), a physician reviewing for board recertification (augmentation), or a layperson seeking diagnostic guidance (consumer).",
    "confidence": "medium"
}


###Example 5
 
O*NET occupation: Mathematical Science Teachers, Postsecondary
O*NET task: Prepare and deliver lectures to undergraduate or graduate students on topics such as linear algebra, differential equations, and discrete mathematics.
 
Conversation:
USER: I'm learning linear algebra, specifically Gaussian Elimination. Please give me a short but detailed summary of based on the following chapter for my math notes!
 
5.4 Special Cases
So far, we have only encountered systems of equations which have exactly one solution. This is not always the case, as is demonstrated in
5x + 2y = 4, 10x + 4y = 13.
A normal attempt to solve this system would be to multiply the top equation by -2 and add to the second equation. This gives 0 = 5...
 
A: Here's a summary of Section 5.4 — Special Cases in Gaussian Elimination:
 
When applying Gaussian Elimination, not all systems have a unique solution. Two special cases arise: inconsistent systems (no solution) where elimination produces a contradiction like 0 = 5, and dependent systems (infinitely many solutions) where elimination produces 0 = 0...
 
Output:
{
    "interaction_type": "education",
    "task_match": "partial",
    "label": "LT0",
    "lt1_reason": null,
    "transferred_from": "none",
    "transferred_from_other": null,
    "rationale": "The user explicitly states 'I'm learning linear algebra' and wants a summary 'for my math notes' \u2014 this is a student doing coursework. The O*NET task is about delivering lectures on linear algebra, which matches the topic but not the user's role (student, not teacher).",
    "confidence": "high"
}
 
###Example 6

O*NET occupation: Information Security Analysts
O*NET task: Analyze and assess damage to data/infrastructure as a result of security incidents, examine available recovery tools and processes, and recommend solutions.
{
    "interaction_type": "augmentation",
    "task_match": "good",
    "label": "LT0",
    "lt1_reason": null,
    "transferred_from": "none",
    "transferred_from_other": null,
    "rationale": "The user is conducting a smart contract security audit, presenting a detailed vulnerability analysis with code, attack scenario, and a proposed fix \u2014 the workflow of a professional security researcher or blockchain auditor doing their own job, not a layperson seeking a paid security service.",
    "confidence": "high"
}  

Conversation: 
user: confirm if this valid and correct or invalid this vulnerability notifyRewardAmount() function which is intended to sync and distribute new rewards over time.

Here’s the cause of concern in the notifyRewardAmount() function:


function notifyRewardAmount() external updateReward(false) {
    if (msg.sender != address(pirexEth)) revert Errors.NotPirexEth();

    uint256 rewardBalance = asset.balanceOf(address(this)) -
        totalStaked -
        earned();

    rewardRate = rewardBalance / REWARDS_DURATION;

    if (rewardRate == 0) revert Errors.NoRewards();

    lastUpdateTime = block.timestamp;
    periodFinish = block.timestamp + REWARDS_DURATION;

    emit RewardAdded(rewardBalance);
}



### Vulnerability Details

According to the Solidity code provided, the notifyRewardAmount function computes the rewardBalance using the current contract balance, the totalStaked, and the earned rewards. However, if additional tokens (which are not reward tokens or tokens intended for yield farming/staking) are erroneously or maliciously sent to this contract, the computation of rewardBalance could get tainted.

Evidence:

- The contract takes the token balance of this address.
- It then subtracts totalStaked, which implies that it assumes the entire balance should consist of totalStaked plus the new rewards.
- But if someone sends tokens to the contract that aren’t part of the staking or rewards, they would inflate the rewardBalance.

Scenario:

- An attacker or an honest user sends X amount of asset tokens directly to the contract.
- When notifyRewardAmount is subsequently called, the rewardBalance is inflated by X.
- The computed rewardRate would include these extra tokens.
- Over the REWARDS_DURATION, these additional tokens will be wrongly distributed as rewards to stakers, thus abusing the reward mechanism.

### Suggested Fix

One way to prevent such an issue is to have an exclusively dedicated function to add rewards to the contract that performs strict accounting to ensure that only intended reward additions are treated as such.

### Severity and Type

- Severity: High \ud83d\udd25 - Incorrect calculation of the reward balance can lead to an incorrect distribution of rewards, potentially disbursing more than what should be allowed.
- Type: Logical Error / Design Flaw - This vulnerability arises from a logical oversight in the accounting for rewards.

Affected Code:

In the contract provided, the affected code is in the notifyRewardAmount() function where the rewardBalance is calculated.

### Conclusion

Despite my initial response, this deeper inspection reveals that the contract does indeed have a potential logical flaw in the `notifyReward in this contract // SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import { Ownable2Step } from "openzeppelin-contracts/contracts/access/Ownable2Step.sol";
import { ERC4626 } from "solmate/mixins/ERC4626.sol";
import { ERC20 } from "solmate/tokens/ERC20.sol";
import { FixedPointMathLib } from "solmate/utils/FixedPointMathLib.sol";
import { SafeTransferLib } from "solmate/utils/SafeTransferLib.sol";
import { Errors } from "./libraries/Errors.sol";
import { IPirexEth } from "./interfaces/IPirexEth.sol";

/**
 * @title AutoPxEth
 * @notice Autocompounding vault for (staked) pxETH, adapted from pxCVX vault system
 * @dev This contract enables autocompounding for pxETH assets and includes various fee mechanisms.
 * @author redactedcartel.finance
 */
contract AutoPxEth is Ownable2Step, ERC4626 {
    /**
     * @dev Library: SafeTransferLib - Provides safe transfer functions for ERC20 tokens.
     */
    using SafeTransferLib for ERC20;

    /**
     * @dev Library: FixedPointMathLib - Provides fixed-point arithmetic for uint256.
     */
    using FixedPointMathLib for uint256;

    // Constants

    /**
     * @dev Maximum withdrawal penalty percentage.
     */
    uint256 private constant MAX_WITHDRAWAL_PENALTY = 50_000;
    

    /**
     * @dev Maximum platform fee percentage.
     */
    uint256 private constant MAX_PLATFORM_FEE = 200_000;

    /**
     * @dev Fee denominator for precise fee calculations.
     */
    uint256 private constant FEE_DENOMINATOR = 1_000_000;

    /**
     * @dev Duration of the rewards period.
     */
    uint256 private constant REWARDS_DURATION = 7 days;

    // State variables for tracking rewards and actively staked assets

    /**
     * @notice Reference to the PirexEth contract.
     */
    IPirexEth public pirexEth;
    
    /**
     * @notice Timestamp when the current rewards period will end.
     */
    uint256 public periodFinish;
    
    /**
     * @notice Rate at which rewards are distributed per second.
     */
    uint256 public rewardRate;
    
    /**
     * @notice Timestamp of the last update to the reward variables.
     */
    uint256 public lastUpdateTime;
    
    /**
     * @notice Accumulated reward per token stored.
     */
    uint256 public rewardPerTokenStored;
    
    /**
     * @notice Last calculated reward per token paid to stakers.
     */
    uint256 public rewardPerTokenPaid;
    
    /**
     * @notice Total rewards available for distribution.
     */
    uint256 public rewards;
    
    /**
     * @notice Total assets actively staked in the vault.
     */
    uint256 public totalStaked;

    // State variables related to fees
    /**
     * @notice Withdrawal penalty percentage.
     */
    uint256 public withdrawalPenalty = 30_000;
    
    /**
     * @notice Platform fee percentage.
     */
    uint256 public platformFee = 100_000;
    
    /**
     * @notice Address of the platform that receives fees.
     */
    address public platform;

    // Events

    /**
     * @notice Emitted when rewards are harvested and staked.
     * @dev This event is emitted when a user triggers the harvest function.
     * @param caller address indexed Address that triggered the harvest.
     * @param value  uint256         Amount of rewards harvested.
     */
    event Harvest(address indexed caller, uint256 value);

    /**
     * @notice Emitted when the withdrawal penalty is updated.
     * @dev This event is emitted when the withdrawal penalty is modified.
     * @param penalty uint256 New withdrawal penalty percentage.
     */
    event WithdrawalPenaltyUpdated(uint256 penalty);

    /**
     * @notice Emitted when the platform fee is updated.
     * @dev This event is emitted when the platform fee is modified.
     * @param fee uint256 New platform fee percentage.
     */
    event PlatformFeeUpdated(uint256 fee);
    
    /**
     * @notice Emitted when the platform address is updated.
     * @dev This event is emitted when the platform address is modified.
     * @param _platform address New platform address.
     */
    event PlatformUpdated(address _platform);

    /**
     * @notice Emitted when new rewards are added to the vault.
     * @dev This event is emitted when new rewards are added to the vault.
     * @param reward uint256 Amount of rewards added.
     */
    event RewardAdded(uint256 reward);

    /**
     * @notice Emitted when the PirexEth contract address is set.
     * @dev This event is emitted when the PirexEth contract address is set.
     * @param _pirexEth address New PirexEth contract address.
     */
    event SetPirexEth(address _pirexEth);

    // Modifiers
    /**
     * @dev Update reward states modifier
     * @param updateEarned bool Whether to update earned amount so far
     */
    modifier updateReward(bool updateEarned) {
        rewardPerTokenStored = rewardPerToken();
        lastUpdateTime = lastTimeRewardApplicable();

        if (updateEarned) {
            rewards = earned();
            rewardPerTokenPaid = rewardPerTokenStored;
        }
        _;
    }

    /**
     * @dev Contract constructor
     * @param _asset address Asset contract address
     * @param _platform address Platform address
     */
    constructor(
        address _asset,
        address _platform
    ) ERC4626(ERC20(_asset), "Autocompounding Pirex Ether", "apxETH") {
        if (_platform == address(0)) revert Errors.ZeroAddress();

        platform = _platform;
    }

    /*//////////////////////////////////////////////////////////////
                        RESTRICTED FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /**
     * @notice Set the PirexEth contract address
     * @dev Function access restricted to only owner
     * @param _pirexEth address PirexEth contract address
     */
    function setPirexEth(address _pirexEth) external onlyOwner {
        if (_pirexEth == address(0)) revert Errors.ZeroAddress();

        emit SetPirexEth(_pirexEth);

        pirexEth = IPirexEth(_pirexEth);
    }

    /**
     * @notice Set the withdrawal penalty
     * @dev Function access restricted to only owner
     * @param penalty uint256 Withdrawal penalty
     */
    function setWithdrawalPenalty(uint256 penalty) external onlyOwner {
        if (penalty > MAX_WITHDRAWAL_PENALTY) revert Errors.ExceedsMax();

        withdrawalPenalty = penalty;

        emit WithdrawalPenaltyUpdated(penalty);
    }

    /**
     * @notice Set the platform fee
     * @dev Function access restricted to only owner
     * @param fee uint256 Platform fee
     */
    function setPlatformFee(uint256 fee) external onlyOwner {
        if (fee > MAX_PLATFORM_FEE) revert Errors.ExceedsMax();

        platformFee = fee;

        emit PlatformFeeUpdated(fee);
    }

    /**
     * @notice Set the platform
     * @dev Function access restricted to only owner
     * @param _platform address Platform
     */
    function setPlatform(address _platform) external onlyOwner {
        if (_platform == address(0)) revert Errors.ZeroAddress();

        platform = _platform;

        emit PlatformUpdated(_platform);
    }

    /**
     * @notice Notify and sync the newly added rewards to be streamed over time
     * @dev Rewards are streamed following the duration set in REWARDS_DURATION
     */
    function notifyRewardAmount() external updateReward(false) {
        if (msg.sender != address(pirexEth)) revert Errors.NotPirexEth();

        // Rewards transferred directly to this contract are not added to totalStaked
        // To get the rewards w/o relying on a potentially incorrect passed in arg,
        // we can use the difference between the asset balance and totalStaked.
        // Additionally, to avoid re-distributing rewards, deduct the output of `earned`
        uint256 rewardBalance = asset.balanceOf(address(this)) -
            totalStaked -
            earned();

        rewardRate = rewardBalance / REWARDS_DURATION;

        if (rewardRate == 0) revert Errors.NoRewards();

        lastUpdateTime = block.timestamp;
        periodFinish = block.timestamp + REWARDS_DURATION;

        emit RewardAdded(rewardBalance);
    }

    /*//////////////////////////////////////////////////////////////
                                VIEWS
    //////////////////////////////////////////////////////////////*/

    /**
     * @inheritdoc ERC4626
     * @notice Get the amount of available pxETH in the contract
     * @dev Rewards are streamed for the duration set in REWARDS_DURATION
     */
    function totalAssets() public view override returns (uint256) {
        // Based on the current totalStaked and available rewards
        uint256 _totalStaked = totalStaked;
        uint256 _rewards = ((_totalStaked *
            (rewardPerToken() - rewardPerTokenPaid)) / 1e18) + rewards;

        // Deduct the exact reward amount staked (after fees are deducted when calling `harvest`)
        return
            _totalStaked +
            (
                _rewards == 0
                    ? 0
                    : (_rewards - ((_rewards * platformFee) / FEE_DENOMINATOR))
            );
    }

    /**
     * @notice Returns the last effective timestamp of the current reward period
     * @return uint256 Timestamp
     */
    function lastTimeRewardApplicable() public view returns (uint256) {
        return block.timestamp < periodFinish ? block.timestamp : periodFinish;
    }

    /**
     * @notice Returns the amount of rewards per staked token/asset
     * @return uint256 Rewards amount
     */
    function rewardPerToken() public view returns (uint256) {
        if (totalStaked == 0) {
            return rewardPerTokenStored;
        }

        return
            rewardPerTokenStored +
            ((((lastTimeRewardApplicable() - lastUpdateTime) * rewardRate) *
                1e18) / totalStaked);
    }

    /**
     * @notice Returns the earned rewards amount so far
     * @return uint256 Rewards amount
     */
    function earned() public view returns (uint256) {
        return
            ((totalStaked * (rewardPerToken() - rewardPerTokenPaid)) / 1e18) +
            rewards;
    }

    /**
     * @notice Return the amount of assets per 1 (1e18) share
     * @return uint256 Assets
     */
    function assetsPerShare() external view returns (uint256) {
        return previewRedeem(1e18);
    }

    /*//////////////////////////////////////////////////////////////
                            INTERNAL FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /**
     * @dev Internal method to keep track of the total amount of staked token/asset on deposit/mint
     */
    function _stake(uint256 amount) internal updateReward(true) {
        totalStaked += amount;
    }

    /**
     * @dev Internal method to keep track of the total amount of staked token/asset on withdrawal/redeem
     */
    function _withdraw(uint256 amount) internal updateReward(true) {
        totalStaked -= amount;
    }

    /*//////////////////////////////////////////////////////////////
                            ERC4626 OVERRIDES
    //////////////////////////////////////////////////////////////*/

    /**
     * @inheritdoc ERC4626
     * @dev Deduct the specified amount of assets from totalStaked to prepare for transfer to the user
     * @param assets uint256 Assets
     */
    function beforeWithdraw(uint256 assets, uint256) internal override {
        // Perform harvest to make sure that totalStaked is always equal or larger than assets to be withdrawn
        if (assets > totalStaked) harvest();

        _withdraw(assets);
    }

    /**
     * @inheritdoc ERC4626
     * @dev Include the new assets in totalStaked so that rewards can be properly distributed
     * @param assets uint256 Assets
     */
    function afterDeposit(uint256 assets, uint256) internal override {
        _stake(assets);
    }

    /**
     * @inheritdoc ERC4626
     * @dev Preview the amount of assets a user would receive from redeeming shares
     */
    function previewRedeem(
        uint256 shares
    ) public view override returns (uint256) {
        // Calculate assets based on a user's % ownership of vault shares
        uint256 assets = convertToAssets(shares);

        uint256 _totalSupply = totalSupply;

        // Calculate a penalty - zero if user is the last to withdraw.
        uint256 penalty = (_totalSupply == 0 || _totalSupply - shares == 0)
            ? 0
            : assets.mulDivUp(withdrawalPenalty, FEE_DENOMINATOR); // Round up the penalty in favour of the protocol.

        // Redeemable amount is the post-penalty amount
        return assets - penalty;
    }

    /**
     * @inheritdoc ERC4626
     * @notice Preview the amount of shares a user would need to redeem the specified asset amount
     * @dev This modified version takes into consideration the withdrawal fee
     */
    function previewWithdraw(
        uint256 assets
    ) public view override returns (uint256) {
        // Calculate shares based on the specified assets' proportion of the pool
        uint256 shares = convertToShares(assets);

        // Save 1 SLOAD
        uint256 _totalSupply = totalSupply;

        // Factor in additional shares to fulfill withdrawal if user is not the last to withdraw
        return
            (_totalSupply == 0 || _totalSupply - shares == 0)
                ? shares
                : (shares * FEE_DENOMINATOR) /
                    (FEE_DENOMINATOR - withdrawalPenalty);
    }

    /*//////////////////////////////////////////////////////////////
                            MUTATIVE FUNCTIONS
    //////////////////////////////////////////////////////////////*/

    /**
     * @notice Harvest and stake available rewards after distributing fees to the platform
     * @dev This function claims and stakes the available rewards, deducting a fee for the platform.
     */
    function harvest() public updateReward(true) {
        uint256 _rewards = rewards;

        if (_rewards != 0) {
            rewards = 0;

            // Fee for platform
            uint256 feeAmount = (_rewards * platformFee) / FEE_DENOMINATOR;

            // Deduct fee from reward balance
            _rewards -= feeAmount;

            // Claimed rewards should be in pxETH
            asset.safeTransfer(platform, feeAmount);

            // Stake rewards sans fee
            _stake(_rewards);

            emit Harvest(msg.sender, _rewards);
        }
    }

    /**
     * @notice Override transfer logic to trigger direct `initiateRedemption`.
     * @dev This function overrides the standard transfer logic to initiate redemption when transferring to the PirexEth contract.
     * @param to     address Transfer destination
     * @param amount uint256 Amount
     * @return       bool
     */
    function transfer(
        address to,
        uint256 amount
    ) public override returns (bool) {
        super.transfer(to, amount);

        if (to == address(pirexEth)) {
            pirexEth.initiateRedemption(amount, msg.sender, false);
        }

        return true;
    }

    /**
     * @notice Override transferFrom logic to trigger direct `initiateRedemption`.
     * @dev This function overrides the standard transferFrom logic to initiate redemption when transferring from the PirexEth contract.
     * @param from   Address of the transfer origin.
     * @param to     Address of the transfer destination.
     * @param amount Amount of tokens to transfer.
     * @return       A boolean indicating the success of the transfer.
     */
    function transferFrom(
        address from,
        address to,
        uint256 amount
    ) public override returns (bool) {
        super.transferFrom(from, to, amount);

        if (to == address(pirexEth)) {
            pirexEth.initiateRedemption(amount, from, false);
        }

        return true;
    }
}
assistant: The analysis provided does seem valid. The `notifyRewardAmount()` function in this contract calculates the `rewardBalance` based on the difference between the contract's balance of the staking asset and the `totalStaked` plus `earned` amount. The identified problem is that if additional tokens that are not intended to be part of the rewards or staking are sent to the contract directly, this would inflate the balance used to calculate `rewardBalance`, and subsequently the `rewardRate`, leading to more rewards being distributed than intended during the `REWARDS_DURATION`.

This incorrect calculation can indeed result in a severe logical error or design flaw and could potentially be exploited, which makes the vulnerability high severity since it can lead to an incorrect distribution of rewards. Besides, if unchecked, it can have financial implications, creating an unintended arbitrage opportunity for anyone who accidentally or purposely sends additional tokens to the contract.

The suggested fix, to have a dedicated function that manages the addition of rewards and ensures strict accounting, is a good approach to mitigate this risk. This function should ideally only accept tokens that are intended to be part of the reward system and have proper access control to prevent unauthorized invocation.

The recommended course of action is to review the contract's logic and implement appropriate changes to the reward-management system to address this issue. Extra features like a dedicated reward addition function, as mentioned in the suggested fix, or additional checks on the token balance that only account for recognized reward tokens, should be considered. These changes should be tested thoroughly to prevent similar logical oversights.

Additionally, it's worth noting that adjustments to the design and logic should be made carefully and with an understanding of the wider implications to the contract's functionality to maintain the intended behavior while ensuring security.
user: if it's valid prove that with evidence and real value that is correct vulnerability and exist in the contract that i give you 
assistant: Given the nature of smart contracts and the lack of testable code execution here, I cannot run a live test to prove the vulnerability with real values directly within this context. However, I can provide a logical explanation and a hypothetical scenario based on the given contract code to demonstrate the existence of the vulnerability.

### Based on the provided contract code, let's walk through a hypothetical scenario:

1. Assume that `totalStaked` is 1000 tokens and `earned` (cumulative rewards owed to stakers) is 200 tokens. We can pretend that the contract's `asset.balanceOf(address(this))` should rightfully be 1200 tokens to reflect these amounts.
   
2. An external party mistakenly or intentionally transfers another 300 tokens to the contract that are unrelated to the intended rewards. The contract's `asset.balanceOf(address(this))` will now report 1500 tokens.

3. When `notifyRewardAmount()` is called, it calculates `rewardBalance` as follows:

   ```solidity
   uint256 rewardBalance = asset.balanceOf(address(this)) - totalStaked - earned();
   // In theory, based on steps 1 and 2: rewardBalance = 1500 - 1000 - 200 = 300
   ```

4. However, only the initial 200 tokens were intended rewards. The calculated `rewardRate` is thus based on 300 tokens instead of the correct 200 tokens, leading to overstated rewards being distributed over `REWARDS_DURATION`.

5. This means that over the duration of the rewards period, the rewardRate will be higher than it should be, resulting in additional tokens (in this case, 300 - 200 = 100 extra tokens) being distributed to stakers.

### Suggested Fix Validation:

Implementing a dedicated function to handle the addition of rewards would allow the contract to validate and update its internal accounting with the exact amount of intended rewards. For instance, a function like `addReward(uint256 rewardAmount)` could be added where `rewardAmount` is a tightly controlled parameter representing the new rewards to be added to `totalStaked`.

In this function, checks can be implemented to ensure that only the intended reward tokens are considered when updating the reward rate. For example:

```solidity
function addReward(uint256 rewardAmount) external onlyOwner {
   // Ensure that the transfer of rewardAmount tokens to the contract occurs here, 
   // or check balances before and after a trusted operation to obtain the reward tokens
   
   totalStaked += rewardAmount; // Update internal accounting
   // Logic to update rewardRate based on the newly added rewardAmount
   ...
}
```

A call to `addReward` with controlled `rewardAmount` would then correctly reflect the intended reward logic, and this controlled update would not be affected by any unrelated token transfers to the smart contract.

### Conclusion:

The hypothetical scenario and logical deductions from the contract's code demonstrate that the vulnerability pointed out is genuine and suggests that the `notifyRewardAmount` function is using an imprecise way of calculating rewards, which may lead to an incorrect distribution of rewards if extraneous tokens are present in the contract's balance. Implementing the suggested fix would mitigate this logical error by taking into account only the intended rewards. However, actual proofs would require testing the contract's behavior on a blockchain or in a test environment with realistic deployment and transaction simulations.
user: now how we prove this vulenrbaility in this \tMainnet Deployment for the contract  \t0x9Ba021B0a9b958B5E75cE9f6dff97C7eE52cb3E6
assistant: To prove the existence of the vulnerability on the Ethereum Mainnet for a deployed contract, you would typically follow a multi-step verification process that involves analyzing transactions and potentially setting up a simulation of the contract's execution. However, the actual state and transactions on the Mainnet should be assessed by developers and security auditors with proper tools.

These are the general steps a security auditor or developer might take (note that this is a simplified description, and real-world analysis would require sophisticated tools and expertise):

1. **Smart Contract Review**: Review the deployed contract's source code to ensure that it matches the version provided to you. Developers usually publish verified source code on Etherscan, which is associated with the deployed bytecode.

2. **Historical Transaction Analysis**: Examine past transactions to the contract (using tools like Etherscan) to see if there are any instances where the balance of the contract was inflated due to external tokens being sent to it.

3. **State and Storage Analysis**: Inspect the current state of the contract, focusing specifically on the `totalStaked`, `earned`, and `asset.balanceOf(address(this))` values. It's important to ascertain whether `asset.balanceOf(address(this))` exceeds the sum of recognised staked and rewarded tokens.

4. **Test Case Simulation**: Before interacting with the Mainnet contract, a test scenario reflecting the vulnerability could be simulated on a testnet or a development environment. Security professionals often use frameworks like Hardhat or Truffle, coupled with plugins like Tenderly or Mythril, to simulate contract execution and test for vulnerabilities.

5. **Transaction Crafting and Sending**: If a potential vulnerability is identified through analysis and you have permission to interact with the contract (you might need to be the owner of the contract or have appropriate authorization), create a transaction that would execute the `notifyRewardAmount()` function and observe the results. This must be done with caution, as it can have real financial implications.

6. **Monitoring for Abnormalities**: Monitor reward-related transactions and staking behavior during the subsequent calls to the `notifyRewardAmount()` function by normal operations of the contract. Any unexpected increases in reward distribution rates could be indicative of the vulnerability being exploited.

When dealing with Mainnet contracts, it is crucial to proceed with care. Abusing a vulnerability on the Mainnet could be considered illegal or unethical. Usually, the aim would be to verify and then responsibly disclose any vulnerabilities to the contract owners or associated parties to get them fixed.

If you are not the contract owner, developer, or security auditor hired to inspect the contract, it is not recommended to attempt to verify the vulnerability yourself, especially on Mainnet. Instead, the best course of action would be to report your findings to the contract owner or their security team for further investigation.

Lastly, if this verification needs to be conducted, please ensure compliance with all legal boundaries and ethical guidelines. If in doubt, seeking legal advice is recommended before proceeding with any security testing or probing of deployed contracts on the Mainnet.
"""


def build(title: str, task: str, conversation: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(examples=EXAMPLES)},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                title=title, task=task, conversation=conversation
            ),
        },
    ]
