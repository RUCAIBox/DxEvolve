from typing import Union, Dict, Type, List
from pydantic import BaseModel, Field

from langchain.tools import BaseTool
import pandas as pd

from tools.Actions import get_action_results, Actions

import requests
import xmltodict
import time
from typing import List, Dict, Union

from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
import ssl


class TLS12Adapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        # 创建 SSL 上下文并禁用旧版本协议
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2  # 只允许 TLS 1.2 及以上
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# 在初始化时创建 session 并添加适配器

class SearchPubMed():
    def __init__(self, api_key: str | None = None, max_retries: int = 100, retry_delay: int = 0):
        """
        初始化PubMed搜索器
        
        Args:
            api_key: NCBI API密钥
            max_retries: 最大重试次数
            retry_delay: 重试延迟(秒)
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        

    def _make_request(self, url: str, params: Dict) -> Union[Dict, None]:
    # 设置代理（根据你的实际代理地址）
        proxies = {
            "http": "http://127.0.0.1:7860",
            "https": "http://127.0.0.1:7860"
        }

        session = requests.Session()
        # session.mount("https://", TLS12Adapter())

        for attempt in range(self.max_retries):
            try:
                response = session.get(
                    url,
                    params=params,
                    # proxies=proxies,
                    # verify=True,  # 或 verify=True
                    timeout=1.2,
                    allow_redirects=True
                )
                if response.status_code == 200:
                    return xmltodict.parse(response.text, disable_entities=False)
            except Exception as e:
                print(f"请求失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return None
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        搜索PubMed文献
        
        Args:
            query: 搜索查询
            max_results: 最大返回结果数
            
        Returns:
            文献详细信息列表
        """
        # 第一步: 获取文献ID
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
        }
        if self.api_key:
            search_params["api_key"] = self.api_key
        
        search_data = self._make_request(f"{self.base_url}esearch.fcgi", search_params)
        if not search_data or 'IdList' not in search_data['eSearchResult'] or search_data['eSearchResult']['IdList'] == None:
            # return [], None
            for retry in range(3):
                time.sleep(10) 
                print(f"Retrying... [{retry+1}/3]...")
                search_data = self._make_request(f"{self.base_url}esearch.fcgi", search_params)
                
                if search_data and 'IdList' in search_data['eSearchResult'] and search_data['eSearchResult']['IdList'] is not None:
                    break
        if not search_data or 'IdList' not in search_data['eSearchResult'] or search_data['eSearchResult']['IdList'] == None:
            return "Not available. Try a different search query."

        id_list = search_data['eSearchResult']['IdList']['Id']
        if not id_list:
            return "Not available. Try a different search query."
        
        # 第二步: 获取文献详情
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
        }
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        
        fetch_data = self._make_request(f"{self.base_url}efetch.fcgi", fetch_params)
        if not fetch_data:
            return "Not available. Try a different search query."
        
        articles = []
        result_text = f""
        pubmed_article_set = fetch_data.get('PubmedArticleSet', {})
        pubmed_articles = pubmed_article_set.get('PubmedArticle', [])
        
        # 确保总是列表形式
        if not isinstance(pubmed_articles, list):
            pubmed_articles = [pubmed_articles]
        
        for i, article in enumerate(pubmed_articles, 1):
            medline_citation = article.get('MedlineCitation', {})
            article_data = medline_citation.get('Article', {})
            
            # 提取标题
            title = article_data.get('ArticleTitle', 'No title')
            
            # 提取摘要
            abstract = "No abstract"
            if 'Abstract' in article_data and 'AbstractText' in article_data['Abstract']:
                abstract_text = article_data['Abstract']['AbstractText']
                if isinstance(abstract_text, list):
                    abstract = " ".join([item.get('#text', '') if isinstance(item, dict) else str(item) for item in abstract_text])
                elif isinstance(abstract_text, dict):
                    abstract = abstract_text.get('#text', 'No abstract')
                else:
                    abstract = str(abstract_text)
            
            # 提取其他信息
            pmid = medline_citation.get('PMID', {}).get('#text', '未知ID')
            authors = []
            if 'AuthorList' in article_data and 'Author' in article_data['AuthorList']:
                author_list = article_data['AuthorList']['Author']
                if isinstance(author_list, list):
                    authors = [f"{author.get('ForeName', '')} {author.get('LastName', '')}".strip() 
                              for author in author_list if isinstance(author, dict)]
                elif isinstance(author_list, dict):
                    authors = [f"{author_list.get('ForeName', '')} {author_list.get('LastName', '')}".strip()]
            
            journal_info = article_data.get('Journal', {})
            journal = journal_info.get('Title', '未知期刊')
            pub_date = journal_info.get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '未知年份')
            
            articles.append({
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'year': year
            })
            result_text += f"Article {i}: {title}\n"
            result_text += f"   Abstract: {abstract}\n\n"
        
        return result_text
    
    def search_and_format(self, query: str, max_results: int = 10) -> str:
        """
        搜索PubMed文献并返回格式化文本
        
        Args:
            query: 搜索查询
            max_results: 最大返回结果数
            
        Returns:
            格式化后的文献信息文本
        """
        articles = self.search(query, max_results)
        if not articles:
            return "未找到相关文献"
        
        formatted_text = f"找到 {len(articles)} 篇相关文献:\n\n"
        for i, article in enumerate(articles, 1):
            formatted_text += f"{i}. {article['title']}\n"
            formatted_text += f"   作者: {', '.join(article['authors']) if article['authors'] else '未知作者'}\n"
            formatted_text += f"   期刊: {article['journal']}, {article['year']}\n"
            formatted_text += f"   PMID: {article['pmid']}\n"
            formatted_text += f"   摘要: {article['abstract'][:200]}...\n\n"
        
        return formatted_text

class PubMedSearchInput(BaseModel):
    action_input: str = Field(description="The search query for PubMed.")

class SearchPubMedTool(BaseTool):
    name: str = "PubMed Search"
    description: str = "Search PubMed for medical literature and research articles. Useful when you need to find up-to-date medical information, clinical guidelines, or research evidence to support diagnosis."
    args_schema: Type[BaseModel] = PubMedSearchInput
    pubmed_searcher: SearchPubMed = None

    def __init__(self, pubmed_searcher: SearchPubMed, **kwargs):
        super().__init__(**kwargs)
        self.pubmed_searcher = pubmed_searcher

    def _run(self, action_input: str) -> str:
        return self.pubmed_searcher.search(action_input, max_results=3)

    async def _arun(self, action_input: str) -> str:
        return self._run(action_input)
        

class LaboratoryTests_Input(BaseModel):
    action_input: List = Field(description="The names of the lab tests to run.")


class Imaging_Input(BaseModel):
    action_input: Dict = Field(
        description="The names of the imaging modality and region to be scanned."
    )


class RunLaboratoryTests(BaseTool):
    name: str = "Laboratory Tests"
    description: str = "Laboratory Tests. The specific tests must be specified in the 'Action Input' field."
    args_schema: Type[BaseModel] = LaboratoryTests_Input
    action_results: Dict = {}
    lab_test_mapping_df: pd.DataFrame = None
    include_ref_range: bool = False
    bin_lab_results: bool = False

    def _run(self, action_input: List[Union[int, str]]) -> str:
        return get_action_results(
            action=Actions.Laboratory_Tests,
            action_results=self.action_results,
            action_input=action_input,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=self.include_ref_range,
            bin_lab_results=self.bin_lab_results,
        )

    async def _arun(self, action_input: List[Union[int, str]]) -> str:
        return get_action_results(
            action=Actions.Laboratory_Tests,
            action_results=self.action_results,
            action_input=action_input,
            lab_test_mapping_df=self.lab_test_mapping_df,
            include_ref_range=self.include_ref_range,
            bin_lab_results=self.bin_lab_results,
        )


class RunImaging(BaseTool):
    name: str = "Imaging"
    description: str = "Imaging. Scan region AND modality must be specified in the 'Action Input' field."
    args_schema: Type[BaseModel] = Imaging_Input
    action_results: Dict = {}
    already_requested_scans: Dict = {}

    def _run(self, action_input: Dict) -> str:
        return get_action_results(
            action=Actions.Imaging,
            action_results=self.action_results,
            action_input=action_input,
            already_requested_scans=self.already_requested_scans,
        )

    async def _arun(self, action_input: Dict) -> str:
        return get_action_results(
            action=Actions.Imaging,
            action_results=self.action_results,
            action_input=action_input,
            already_requested_scans=self.already_requested_scans,
        )


class DoPhysicalExamination(BaseTool):
    name: str = "Physical Examination"
    description: str = (
        "Perform physical examination of patient and return observations."
    )
    action_results: Dict = {}

    def _run(self, action_input: str) -> str:
        return get_action_results(
            action=Actions.Physical_Examination, action_results=self.action_results
        )

    async def _arun(self, action_input: str) -> str:
        return get_action_results(
            action=Actions.Physical_Examination, action_results=self.action_results
        )


class ReadDiagnosticCriteria(BaseTool):
    name: str = "Diagnostic Criteria"
    description: str = "Read diagnostic criteria for a given pathology."

    def _run(self, action_input: str) -> str:
        return get_action_results(
            action=Actions.Diagnostic_Criteria,
            action_input=action_input,
        )

    async def _arun(self, action_input: str) -> str:
        return get_action_results(
            action=Actions.Diagnostic_Criteria,
            action_input=action_input,
        )
