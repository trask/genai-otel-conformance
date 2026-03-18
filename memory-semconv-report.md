# Memory Semantic Conventions (PR #3250) — Framework Captureability Report

## Framework-Level Data Availability (What *Could* Be Captured)

This table shows which proposed attributes are **available from the framework/SDK API** and could theoretically be set by instrumentation:

### `update_memory`

| Attribute | Mem0 | CrewAI | Bedrock AgentCore | Google ADK | Azure AI Foundry |
|---|---|---|---|---|---|
| `gen_ai.provider.name` | Yes — `mem0` | Yes — `crewai` | Yes — `aws.bedrock` | Yes — `google_adk` | Yes — `az.ai.foundry` |
| `gen_ai.operation.name` | Yes — `update_memory` | Yes — `update_memory` | Yes — `update_memory` | Yes — `update_memory` | Yes — `update_memory` |
| `gen_ai.agent.id` | Yes — `agent_id` | No | No | No | No |
| `gen_ai.conversation.id` | Yes — `run_id` | No | No | Yes — `session.id` | No |
| `gen_ai.memory.expiration_date` | Yes — `expiration_date` param | No | No | No | No |
| `gen_ai.memory.record.content` | Yes — `messages` param | Yes — `content` param | Yes — `content.text` | Yes — session events | Yes — `items` param |
| `gen_ai.memory.record.id` | Yes — `id` in response | No | Yes — `memoryRecordId` in response | No | Yes — `memory_id` in response |
| `gen_ai.memory.store.id` | No (managed service) | No (implicit) | Yes — `memoryId` | No (implicit) | No (uses `name` as identifier) |
| `gen_ai.memory.store.name` | No | No | Yes — `name` from Memory | No | Yes — `name` param |
| `server.address` | Yes — host URL | No (local) | Yes — endpoint URL | No (local) | Yes — endpoint URL |
| `server.port` | Yes — host URL | No (local) | Yes — endpoint URL | No (local) | Yes — endpoint URL |

### `search_memory`

| Attribute | Mem0 | CrewAI | Bedrock AgentCore | Google ADK | Azure AI Foundry |
|---|---|---|---|---|---|
| `gen_ai.provider.name` | Yes — `mem0` | Yes — `crewai` | Yes — `aws.bedrock` | Yes — `google_adk` | Yes — `az.ai.foundry` |
| `gen_ai.operation.name` | Yes — `search_memory` | Yes — `search_memory` | Yes — `search_memory` | Yes — `search_memory` | Yes — `search_memory` |
| `gen_ai.agent.id` | Yes — `agent_id` | No | No | No | No |
| `gen_ai.conversation.id` | Yes — `run_id` | No | No | No | No |
| `gen_ai.memory.query.text` | Yes — `query` param | Yes — `query` param | Yes — `searchQuery` in searchCriteria | Yes — `query` param | Yes — `items` param |
| `gen_ai.memory.search.result.count` | Yes — `len(results)` | Yes — `len(results)` from recall | Yes — `len(memoryRecordSummaries)` | Yes — `len(memories)` | Yes — `len(memories)` |
| `gen_ai.memory.search.similarity.threshold` | Yes — `threshold` param | No | No | No | No |
| `gen_ai.memory.store.id` | No | No | Yes — `memoryId` | No (implicit) | No (uses `name` as identifier) |
| `gen_ai.memory.store.name` | No | No | Yes — `name` from Memory | No | Yes — `name` param |
| `server.address` | Yes | No (local) | Yes | No (local) | Yes |
| `server.port` | Yes | No (local) | Yes | No (local) | Yes |

### `delete_memory`

| Attribute | Mem0 | CrewAI | Bedrock AgentCore | Google ADK | Azure AI Foundry |
|---|---|---|---|---|---|
| Operation exists? | Yes — `delete`/`delete_all` | Yes — `forget`/`reset` | Yes — `batch_delete_memory_records` | No (no delete API) | Yes — `delete_scope` |
| `gen_ai.provider.name` | Yes — `mem0` | Yes — `crewai` | Yes — `aws.bedrock` | — | Yes — `az.ai.foundry` |
| `gen_ai.operation.name` | Yes — `delete_memory` | Yes — `delete_memory` | Yes — `delete_memory` | — | Yes — `delete_memory` |
| `gen_ai.agent.id` | Yes — `agent_id` | No | No | — | No |
| `gen_ai.conversation.id` | Yes — `run_id` | No | No | — | No |
| `gen_ai.memory.record.id` | Yes — `memory_id` on delete | No | Yes — `memoryRecordId` | — | No (scope-level delete) |
| `gen_ai.memory.scope` | Yes — `user_id` on delete_all | Yes — `scope` on forget | No | — | Yes — `scope` param |
| `gen_ai.memory.store.id` | No | No | Yes — `memoryId` | — | No (uses `name` as identifier) |
| `gen_ai.memory.store.name` | No | No | Yes — `name` from Memory | — | Yes — `name` param |
| `server.address` | Yes | No (local) | Yes | — | Yes |
| `server.port` | Yes | No (local) | Yes | — | Yes |

### `create_memory_store`

| Attribute | Mem0 | CrewAI | Bedrock AgentCore | Google ADK | Azure AI Foundry |
|---|---|---|---|---|---|
| Operation exists? | No (managed service) | No (implicit) | Yes — `create_memory` (control plane) | No (implicit) | Yes — `memory_stores.create` |
| `gen_ai.provider.name` | — | — | Yes — `aws.bedrock` | — | Yes — `az.ai.foundry` |
| `gen_ai.operation.name` | — | — | Yes — `create_memory_store` | — | Yes — `create_memory_store` |
| `gen_ai.memory.scope` | — | — | No | — | Yes — from context |
| `gen_ai.memory.expiration_date` | — | — | Yes — `eventExpiryDuration` in CreateMemoryInput | — | No |
| `gen_ai.memory.store.id` | — | — | Yes — returned `memoryId` | — | Yes — `id` in response |
| `gen_ai.memory.store.name` | — | — | Yes — `name` in CreateMemoryInput | — | Yes — `name` param |
| `server.address` | — | — | Yes | — | Yes |
| `server.port` | — | — | Yes | — | Yes |

### `delete_memory_store`

| Attribute | Mem0 | CrewAI | Bedrock AgentCore | Google ADK | Azure AI Foundry |
|---|---|---|---|---|---|
| Operation exists? | No (managed service) | No (implicit) | Yes — `delete_memory` (control plane) | No | Yes — `memory_stores.delete` |
| `gen_ai.provider.name` | — | — | Yes — `aws.bedrock` | — | Yes — `az.ai.foundry` |
| `gen_ai.operation.name` | — | — | Yes — `delete_memory_store` | — | Yes — `delete_memory_store` |
| `gen_ai.memory.store.id` | — | — | Yes — `memoryId` param | — | No (uses `name` as identifier) |
| `gen_ai.memory.store.name` | — | — | Yes — `name` from Memory | — | Yes — `name` param |
| `server.address` | — | — | Yes | — | Yes |
| `server.port` | — | — | Yes | — | Yes |
